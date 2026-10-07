"""Task catalog: parameter schemas for the wizard + safe argv construction.

The schemas below were written against the real CLIs of the underlying
scripts (argparse blocks in each file):

- inference:  /home/MMDE/mmde/methods/run_<method>.py (one script per method,
  all taking --dataset/--split plus per-script extras). The methods list is
  discovered by globbing run_*.py; the whole inference entry is dropped when
  the methods dir is missing. Optional flags are only emitted when the
  selected method's script actually declares them (source scan), so a flag
  one script lacks never breaks another.
- eval:       /home/MMDE/mmde/eval/eval_mmde.py
  (--dataset, --split, --models nargs+, --protocol mmde|standard).
- fuse_scene: /home/MMDE/mmde/scripts/fuse_scene.py (--dataset/--split/
  --model, --voxel/--stride/--min-depth/--max-depth/--max-frames/--force).

Security: params are a JSON object; only whitelisted ParamSpec fields are
accepted, name-like values (dataset/split/model/...) must match NAME_RE, and
the command line is always built as an argv list (never a shell string).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..registry import get_registry
from ..routers.common import NAME_RE

log = logging.getLogger(__name__)

# External script locations and the Python interpreter used to run tasks.
# All overridable via env vars so a migrated deployment only needs to set
# environment instead of patching code:
#   MMDE_PYTHON        — interpreter for inference/eval/fusion subprocesses
#   MMDE_METHODS_DIR   — dir containing run_<method>.py inference wrappers
#                        (default: <studio root>/methods, next to backend/)
#   MMDE_EVAL_SCRIPT   — eval_mmde.py path
#   MMDE_FUSE_SCRIPT   — fuse_scene.py path
# Per-method interpreters for the heavy lifting live in
# methods/interpreters.yaml (machine-local config); the wrappers themselves
# only need the backend deps.
DGGT_PYTHON = os.environ.get("MMDE_PYTHON", "/root/miniconda3/envs/dggt/bin/python")
METHODS_DIR = Path(os.environ.get(
    "MMDE_METHODS_DIR", Path(__file__).resolve().parents[3] / "methods"))
EVAL_SCRIPT = Path(os.environ.get("MMDE_EVAL_SCRIPT", "/home/MMDE/mmde/eval/eval_mmde.py"))
FUSE_SCRIPT = Path(os.environ.get("MMDE_FUSE_SCRIPT", "/home/MMDE/mmde/scripts/fuse_scene.py"))

_FLAG_RE = re.compile(r"""add_argument\(\s*["'](-{1,2}[a-zA-Z0-9-]+)""")


class ParamError(ValueError):
    """Invalid task params; mapped to HTTP 400 by the router."""


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    name: str                     # key in the params JSON object
    label: str                    # wizard display label
    kind: str                     # str | int | float | bool | list
    flag: str | None = None       # CLI flag; None = selector (not passed on)
    required: bool = False
    default: Any = None
    choices: tuple | Callable[[dict], tuple] | None = None
    help: str = ""
    name_kind: bool = False       # NAME_RE-validated (no path injection)
    dynamic_choices: bool = False  # resolved per dataset/split; catalog emits None


@dataclass(frozen=True)
class TaskSpec:
    type: str
    title: str
    description: str
    params: tuple
    script: str | Callable[[dict], str]
    interpreter: str | None = DGGT_PYTHON
    cwd: str | None = None
    env: tuple = ()               # extra (key, value) env pairs
    filter_flags: bool = False    # emit only flags the script declares


def _bad_name(s: str) -> bool:
    return not NAME_RE.match(s) or s in (".", "..")


def _coerce(p: ParamSpec, raw: Any) -> Any:
    if p.kind == "str":
        if not isinstance(raw, (str, int, float)):
            raise ParamError(f"param {p.name}: expected string")
        return str(raw).strip()
    if p.kind == "int":
        if isinstance(raw, bool):
            raise ParamError(f"param {p.name}: expected int")
        return int(raw)
    if p.kind == "float":
        if isinstance(raw, bool):
            raise ParamError(f"param {p.name}: expected number")
        return float(raw)
    if p.kind == "bool":
        if not isinstance(raw, bool):
            raise ParamError(f"param {p.name}: expected bool")
        return raw
    if p.kind == "list":
        if isinstance(raw, str):
            items = [s.strip() for s in raw.split(",") if s.strip()]
        elif isinstance(raw, list):
            items = [str(s).strip() for s in raw]
        else:
            raise ParamError(f"param {p.name}: expected list or comma string")
        return items
    raise ParamError(f"param {p.name}: unknown kind {p.kind}")


def _resolve_choices(p: ParamSpec, params: dict) -> tuple | None:
    if p.choices is None:
        return None
    if callable(p.choices):
        try:
            return tuple(p.choices(params))
        except Exception:
            log.warning("choices for param %s failed to resolve", p.name)
            return ()
    return tuple(p.choices)


def validate_params(spec: TaskSpec, params: Any) -> dict:
    """Whitelist-validate and coerce a params object against a TaskSpec."""
    if not isinstance(params, dict):
        raise ParamError("params must be a JSON object")
    known = {p.name for p in spec.params}
    extra = sorted(set(params) - known)
    if extra:
        raise ParamError(f"unknown params: {', '.join(extra)}")
    out: dict = {}
    for p in spec.params:
        raw = params.get(p.name)
        if raw is None or raw == "":
            if p.required:
                raise ParamError(f"missing required param: {p.name}")
            out[p.name] = p.default
            continue
        try:
            v = _coerce(p, raw)
        except (TypeError, ValueError):
            raise ParamError(
                f"param {p.name}: expected {p.kind}") from None
        values = v if p.kind == "list" else [v]
        for item in values:
            if p.name_kind and _bad_name(str(item)):
                raise ParamError(f"param {p.name}: invalid name {item!r}")
        choices = _resolve_choices(p, out)
        if choices is not None and values:
            bad = [str(i) for i in values if str(i) not in choices]
            if bad:
                raise ParamError(
                    f"param {p.name}: not allowed: {', '.join(bad)}")
        out[p.name] = v
    return out


def _script_flags(script: str) -> set[str]:
    try:
        text = Path(script).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    return set(_FLAG_RE.findall(text))


def spec_script(spec: TaskSpec, params: dict) -> str:
    script = spec.script(params) if callable(spec.script) else spec.script
    return script


def build_argv(spec: TaskSpec, params: dict) -> list[str]:
    """Expand validated params into an argv list (script first, no shell)."""
    script = spec_script(spec, params)
    argv = [script]
    supported = _script_flags(script) if spec.filter_flags else None
    for p in spec.params:
        if p.flag is None:
            continue  # selector param (e.g. inference method)
        v = params.get(p.name)
        if v is None:
            continue
        if supported is not None and p.flag not in supported:
            continue
        if p.kind == "bool":
            if v:
                argv.append(p.flag)
        elif p.kind == "list":
            if v:
                argv.append(p.flag)
                argv += [str(x) for x in v]
        else:
            argv += [p.flag, str(v)]
    return argv


def spec_to_json(spec: TaskSpec) -> dict:
    params = []
    for p in spec.params:
        d = {"name": p.name, "label": p.label, "type": p.kind,
             "required": p.required, "default": p.default, "help": p.help}
        if p.dynamic_choices:
            d["choices"] = None
        elif p.choices is not None:
            d["choices"] = list(_resolve_choices(p, {}) or ())
        params.append(d)
    return {"type": spec.type, "title": spec.title,
            "description": spec.description, "params": params}


# ---------------------------------------------------------------------------
# dynamic choice sources
# ---------------------------------------------------------------------------

def _dataset_choices(_params: dict) -> tuple:
    return tuple(get_registry().names())


def _fuse_model_choices(params: dict) -> tuple:
    entry = get_registry().get_dataset(str(params.get("dataset") or ""))
    info = get_registry().splits(entry).get(str(params.get("split") or ""))
    return tuple(info.models) if info else ()


def _discover_methods() -> tuple:
    if not METHODS_DIR.is_dir():
        return ()
    return tuple(sorted(p.stem[len("run_"):]
                        for p in METHODS_DIR.glob("run_*.py")))


# ---------------------------------------------------------------------------
# built-in specs (availability follows script existence)
# ---------------------------------------------------------------------------

def _inference_spec(methods: tuple) -> TaskSpec:
    return TaskSpec(
        type="inference",
        title="Depth inference",
        description="Run an MMDE method (mmde/methods/run_<method>.py) over "
                    "a dataset split and write predictions to the preds tree.",
        script=lambda p: str(METHODS_DIR / f"run_{p['method']}.py"),
        filter_flags=True,
        params=(
            ParamSpec("method", "Method", "str", required=True,
                      choices=methods, name_kind=True,
                      help="run_*.py method name from mmde/methods"),
            ParamSpec("dataset", "Dataset", "str", "--dataset",
                      required=True, choices=_dataset_choices,
                      name_kind=True, help="registry dataset name"),
            ParamSpec("split", "Split", "str", "--split",
                      required=True, name_kind=True,
                      help="split name (e.g. test_single / test_sequence)"),
            ParamSpec("max_frames", "Max frames", "int", "--max-frames",
                      help="smoke test: process only the first N frames"),
            ParamSpec("model_name", "Model name", "str", "--model-name",
                      name_kind=True,
                      help="prediction subdir name (script default if unset)"),
            ParamSpec("device", "Device", "str", "--device", name_kind=True,
                      help="cuda / cpu (script default if unset)"),
        ),
    )


def _eval_spec() -> TaskSpec:
    return TaskSpec(
        type="eval",
        title="Evaluation",
        description="Run eval_mmde.py: per-frame metrics of prediction "
                    "models against GT on a dataset split.",
        script=str(EVAL_SCRIPT),
        params=(
            ParamSpec("dataset", "Dataset", "str", "--dataset",
                      required=True, choices=_dataset_choices,
                      name_kind=True, help="registry dataset name"),
            ParamSpec("split", "Split", "str", "--split",
                      help="split name (script default: test_single)"),
            ParamSpec("models", "Models", "list", "--models",
                      name_kind=True,
                      help="comma-separated prediction model names "
                           "(script default: dav2_rel, unidepth_v2, moge2)"),
            ParamSpec("protocol", "Protocol", "str", "--protocol",
                      default="mmde", choices=("mmde", "standard"),
                      help="mmde: legacy full-frame 200 m; standard: "
                           "documented per-dataset protocol"),
        ),
    )


def _fuse_spec() -> TaskSpec:
    return TaskSpec(
        type="fuse_scene",
        title="Scene fusion",
        description="Run fuse_scene.py: fuse per-frame depth predictions "
                    "into the chunked scene served by /api/scene/*. Use "
                    "--force to overwrite an existing scene dir.",
        script=str(FUSE_SCRIPT),
        params=(
            ParamSpec("dataset", "Dataset", "str", "--dataset",
                      required=True, choices=_dataset_choices,
                      name_kind=True, help="registry dataset name"),
            ParamSpec("split", "Split", "str", "--split",
                      required=True, name_kind=True, help="split name"),
            ParamSpec("model", "Model", "str", "--model",
                      name_kind=True, dynamic_choices=True,
                      choices=_fuse_model_choices,
                      help="prediction model to fuse (must exist in the "
                           "preds tree for the dataset/split)"),
            ParamSpec("voxel", "Voxel size", "float", "--voxel",
                      help="voxel downsample size, m"),
            ParamSpec("stride", "Frame stride", "int", "--stride",
                      help="use every Nth frame"),
            ParamSpec("min_depth", "Min depth", "float", "--min-depth"),
            ParamSpec("max_depth", "Max depth", "float", "--max-depth"),
            ParamSpec("max_frames", "Max frames", "int", "--max-frames",
                      help="smoke test: fuse only the first N frames"),
            ParamSpec("force", "Force overwrite", "bool", "--force",
                      help="overwrite an existing complete scene dir"),
        ),
    )


def base_specs() -> list[TaskSpec]:
    specs: list[TaskSpec] = []
    methods = _discover_methods()
    if methods:
        specs.append(_inference_spec(methods))
    if EVAL_SCRIPT.is_file():
        specs.append(_eval_spec())
    if FUSE_SCRIPT.is_file():
        specs.append(_fuse_spec())
    return specs


# ---------------------------------------------------------------------------
# registry (extras: test-only specs, never registered in production)
# ---------------------------------------------------------------------------

_extras: dict[str, TaskSpec] = {}


def register_extra(spec: TaskSpec) -> None:
    _extras[spec.type] = spec


def clear_extras() -> None:
    _extras.clear()


def all_specs() -> list[TaskSpec]:
    return base_specs() + list(_extras.values())


def get_spec(type_: str) -> TaskSpec | None:
    for spec in base_specs():
        if spec.type == type_:
            return spec
    return _extras.get(type_)
