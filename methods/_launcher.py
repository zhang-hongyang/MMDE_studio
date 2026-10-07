"""Small, dependency-light launcher shared by MMDE method wrappers."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

METHODS_DIR = Path(__file__).resolve().parent
STUDIO_ROOT = METHODS_DIR.parent
INTERPRETERS = METHODS_DIR / "interpreters.yaml"
ADAPTER_ROOT = STUDIO_ROOT / "algorithms" / "mmde-adapters"


def datasets_config() -> Path:
    return Path(os.environ.get("MMDE_STUDIO_DATASETS") or
                STUDIO_ROOT / "backend" / "config" / "datasets.yaml")


def dataset_entry(dataset: str) -> dict:
    cfg = datasets_config()
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    try:
        return doc["datasets"][dataset]
    except KeyError:
        raise SystemExit(f"dataset {dataset!r} not in {cfg}")


def resolve_interpreter(key: str) -> str:
    env_key = f"MMDE_{key.upper()}_PYTHON"
    env = os.environ.get(env_key)
    if env:
        return env
    if INTERPRETERS.is_file():
        doc = yaml.safe_load(INTERPRETERS.read_text(encoding="utf-8")) or {}
        if doc.get(key):
            return str(doc[key])
    raise SystemExit(
        f"no interpreter configured for {key!r}; set {env_key} or add it "
        f"to {INTERPRETERS}")


def output_dir(dataset: str, split: str, model_name: str) -> Path:
    entry = dataset_entry(dataset)
    return Path(entry["pred_root"]) / split / model_name


def launch(
    *,
    interpreter_key: str,
    worker: str,
    dataset: str,
    split: str,
    model_name: str,
    max_frames: int | None,
    device: str | None,
    extra: list[str] | None = None,
) -> int:
    out = output_dir(dataset, split, model_name)
    out.mkdir(parents=True, exist_ok=True)
    interpreter = resolve_interpreter(interpreter_key)
    cmd = [
        interpreter,
        str(ADAPTER_ROOT / worker),
        "--datasets-config", str(datasets_config()),
        "--dataset", dataset,
        "--split", split,
        "--output-dir", str(out),
    ]
    if max_frames is not None:
        cmd += ["--max-frames", str(max_frames)]
    if device:
        cmd += ["--device", device]
    if extra:
        cmd += extra
    env = dict(os.environ)
    if interpreter_key == "temporal":
        # PTC is linked against the C++ OpenCV/libstdc++ shipped in its conda
        # prefix.  Prefer those ABI-matched libraries without mutating the
        # host's system toolchain.
        prefix_lib = str(Path(interpreter).resolve().parent.parent / "lib")
        current = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = (f"{prefix_lib}:{current}"
                                  if current else prefix_lib)
    print(json.dumps({"method": model_name, "argv": cmd}, ensure_ascii=False),
          flush=True)
    try:
        proc = subprocess.run(cmd, cwd=STUDIO_ROOT, env=env)
    except OSError as exc:
        raise SystemExit(f"failed to launch {interpreter_key}: {exc}")
    return proc.returncode


def finish(code: int) -> None:
    if code:
        raise SystemExit(code)
    raise SystemExit(0)
