"""Config-driven dataset registry with automatic per-split discovery.

Discovery (splits / cameras / control sources / models / scene models / frame
counts) is cached per dataset and invalidated by mtime: any frames.jsonl
mtime change, or a directory mtime change anywhere under the dataset roots,
triggers a rediscovery on next access. All access is thread-safe.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import get_settings

log = logging.getLogger(__name__)

KNOWN_PLATFORMS = {"vehicle", "uav", "satellite"}
# Split-root triangulation dirs double as sparse-control sources; the npz path
# for a frame comes from its frames.jsonl field (see data_access).
_CONTROL_SOURCE_FIELDS = {
    "calib_seq_triangulation": "calib_seq_path",
    "calib_mono_feye_triangulation": "calib_mono_feye_path",
}


class UnknownDatasetError(Exception):
    """404 semantics: dataset name not in the registry."""


class UnknownSplitError(Exception):
    """404 semantics: split not discovered for this dataset."""


_taxonomy_cache: dict | None = None


def get_model_taxonomy() -> dict:
    """Model classification rules for the frontend dropdowns (models.yaml).

    Cached in-process; reloads when the file mtime changes. Falls back to a
    single catch-all category when the file is missing/unreadable so the UI
    degrades to a flat list instead of breaking.
    """
    global _taxonomy_cache
    from .config import get_settings
    path = get_settings().models_config_path
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {"excluded": [], "categories": [
            {"id": "all", "label": "模型", "match": ["*"]}]}
    if _taxonomy_cache and _taxonomy_cache.get("_mtime") == mtime:
        return _taxonomy_cache
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = {
        "excluded": list(data.get("excluded") or []),
        "categories": list(data.get("categories") or []),
    }
    data["_mtime"] = mtime
    _taxonomy_cache = data
    return data


def reset_taxonomy_cache() -> None:
    global _taxonomy_cache
    _taxonomy_cache = None


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    platform: str
    title: str
    test_root: Path | None
    pred_root: Path | None
    metrics_root: Path | None
    scene_root: Path | None
    calib_protocols: tuple = ()


@dataclass
class SplitInfo:
    name: str
    test_type: str = "test_single"
    n_frames: int = 0
    evaluation_available: bool = True
    cameras: list = field(default_factory=list)
    control_sources: list = field(default_factory=list)
    models: list = field(default_factory=list)
    model_availability: dict = field(default_factory=dict)
    scene_models: list = field(default_factory=list)


def _existing(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_dir() else None


class Registry:
    """Loaded from yaml; per-dataset SplitInfo discovery with mtime cache."""

    _instance: "Registry | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "Registry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(get_settings().config_path)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.summary_path: Path | None = None
        self._datasets: dict[str, DatasetEntry] = {}
        # name -> (signature, {split: SplitInfo})
        self._split_cache: dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        sp = doc.get("summary_path")
        self.summary_path = Path(sp) if sp else None
        for name, cfg in (doc.get("datasets") or {}).items():
            if not isinstance(cfg, dict):
                log.warning("dataset %r: config entry is not a mapping, skipped", name)
                continue
            platform = str(cfg.get("platform", "vehicle"))
            if platform not in KNOWN_PLATFORMS:
                # validate but pass through unknown values
                log.warning("dataset %r: unknown platform %r", name, platform)
            test_root = _existing(cfg.get("test_root"))
            if test_root is None:
                log.warning("dataset %r: missing/unreadable test_root %r, skipped",
                            name, cfg.get("test_root"))
                continue
            self._datasets[name] = DatasetEntry(
                name=name,
                platform=platform,
                title=str(cfg.get("title", name)),
                test_root=test_root,
                pred_root=_existing(cfg.get("pred_root")),
                metrics_root=_existing(cfg.get("metrics_root")),
                scene_root=_existing(cfg.get("scene_root")),
                calib_protocols=tuple(cfg.get("calib_protocols") or ()),
            )

    # -- access -----------------------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._datasets)

    def get_dataset(self, name: str) -> DatasetEntry:
        try:
            return self._datasets[name]
        except KeyError:
            raise UnknownDatasetError(name) from None

    def splits(self, entry: DatasetEntry) -> dict[str, SplitInfo]:
        """Discovered split infos, cached until the dataset's signature changes."""
        sig = self._signature(entry)
        with self._lock:
            cached = self._split_cache.get(entry.name)
            if cached and cached[0] == sig:
                return cached[1]
        discovered = self._discover(entry)
        with self._lock:
            self._split_cache[entry.name] = (sig, discovered)
        return discovered

    def split_info(self, entry: DatasetEntry, split: str) -> SplitInfo:
        info = self.splits(entry).get(split)
        if info is None:
            raise UnknownSplitError(f"{entry.name}/{split}")
        return info

    # -- discovery --------------------------------------------------------

    def _signature(self, entry: DatasetEntry) -> tuple:
        """Mtimes of everything discovery depends on (cheap stat sweep)."""
        parts: list[tuple[str, int]] = [("config", self._mtime(self.config_path))]
        roots = [("test", entry.test_root), ("pred", entry.pred_root),
                 ("scene", entry.scene_root)]
        for tag, root in roots:
            if root is None:
                parts.append((tag, -1))
                continue
            parts.append((tag, self._mtime(root)))
            try:
                children = sorted(p for p in root.iterdir() if p.is_dir())
            except OSError:
                children = []
            for d in children:
                parts.append((f"{tag}:{d.name}", self._mtime(d)))
                if tag != "test":
                    continue
                # one level deeper: split internals that affect discovery
                for sub_name in ("sparse_controls",):
                    sub = d / sub_name
                    if sub.is_dir():
                        parts.append((f"{tag}:{d.name}/{sub_name}", self._mtime(sub)))
                        for s in sorted(p for p in sub.iterdir() if p.is_dir()):
                            parts.append((f"{tag}:{d.name}/{sub_name}/{s.name}",
                                          self._mtime(s)))
                # triangulation control dirs live at split root
                for s in sorted(d.glob("calib_*_triangulation")):
                    if s.is_dir():
                        parts.append((f"{tag}:{d.name}/{s.name}", self._mtime(s)))
                fj = d / "frames.jsonl"
                parts.append((f"{tag}:{d.name}/frames.jsonl", self._mtime(fj)))
        return tuple(parts)

    @staticmethod
    def _mtime(p: Path) -> int:
        try:
            return p.stat().st_mtime_ns
        except OSError:
            return -1

    def _discover(self, entry: DatasetEntry) -> dict[str, SplitInfo]:
        out: dict[str, SplitInfo] = {}
        for split_dir in sorted(entry.test_root.iterdir()):
            if not split_dir.is_dir():
                continue
            frames_file = split_dir / "frames.jsonl"
            if not frames_file.is_file():
                continue
            info = SplitInfo(
                name=split_dir.name,
                test_type=("test_sequence"
                           if split_dir.name.startswith("sequence_")
                           else "test_single"),
            )
            cameras: set[str] = set()
            try:
                with open(frames_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        info.n_frames += 1
                        try:
                            fr = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        cam = fr.get("camera")
                        if fr.get("evaluation_available") is False:
                            info.evaluation_available = False
                        if cam:
                            cameras.add(str(cam))
            except OSError:
                pass
            info.cameras = sorted(cameras)
            info.control_sources = self._control_sources(split_dir)
            if entry.pred_root is not None:
                pred_split = entry.pred_root / info.name
                if pred_split.is_dir():
                    info.models = sorted(p.name for p in pred_split.iterdir()
                                         if p.is_dir())
                    info.model_availability = self._model_availability(
                        split_dir, pred_split, info.models,
                        (entry.metrics_root / info.name
                         if entry.metrics_root is not None else None))
            if entry.scene_root is not None:
                scene_split = entry.scene_root / info.name
                if scene_split.is_dir():
                    info.scene_models = sorted(
                        p.name for p in scene_split.iterdir()
                        if p.is_dir() and (p / "index.json").is_file())
            out[info.name] = info
        return out

    @staticmethod
    def _model_availability(split_dir: Path, pred_split: Path,
                            models: list[str], metrics_split: Path | None) -> dict:
        """Merge split protocol metadata with model-output availability."""
        metadata: dict[str, dict] = {}
        split_path = split_dir / "availability.json"
        try:
            split_doc = json.loads(split_path.read_text(encoding="utf-8"))
            for name, value in (split_doc.get("models") or {}).items():
                if isinstance(value, dict):
                    metadata[str(name)] = dict(value)
        except (OSError, ValueError, AttributeError):
            pass
        for model in models:
            model_path = pred_split / model / "availability.json"
            try:
                value = json.loads(model_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    metadata[model] = {**metadata.get(model, {}), **value}
            except (OSError, ValueError):
                pass
            metadata.setdefault(model, {"available": True})
            metadata[model].setdefault("available", True)
            metric_path = metrics_split / f"{model}.json" if metrics_split else None
            try:
                metric = json.loads(metric_path.read_text(encoding="utf-8"))
                if (metric.get("frames_total", 0) > 0 and
                        metric.get("frames_evaluated") == 0 and
                        metric.get("frames_invalid") == metric.get("frames_total")):
                    metadata[model].update({
                        "available": False,
                        "mode": "all_frames_explicitly_invalid",
                        "reason": (
                            f"all {metric['frames_total']} frames are explicit "
                            "invalid sentinels; metrics and fused scene unavailable"
                        ),
                    })
            except (OSError, ValueError, AttributeError):
                pass
        return {model: metadata[model] for model in models}

    @staticmethod
    def _control_sources(split_dir: Path) -> list[str]:
        sources: list[str] = []
        sc = split_dir / "sparse_controls"
        if sc.is_dir():
            sources += sorted(p.name for p in sc.iterdir() if p.is_dir())
        for d in sorted(split_dir.glob("calib_*_triangulation")):
            if d.is_dir() and any(d.rglob("*.npz")):
                sources.append(d.name)
        return sources


def get_registry() -> Registry:
    return Registry.get()


def reset() -> None:
    Registry.reset()
    reset_taxonomy_cache()
