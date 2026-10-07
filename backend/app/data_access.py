"""Data access layer, faithfully ported from mmde/scripts/pc_viewer_server.py.

Binary layouts are byte-for-byte identical to the old service (see docs/API.md).
fit_affine / sample_pred are local pure-numpy implementations: the old service
imported them from Manydepth2 via a sys.path hack, and sample_pred used
cv2.remap, which crashes on 4K images (map coordinates exceed cv2's internal
SHRT_MAX handling for certain border modes). The numpy bilinear sampler below
sidesteps that entirely; out-of-bounds uv returns NaN.
"""
from __future__ import annotations

import io
import json
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .config import get_settings
from .disk_cache import DiskCache
from .registry import DatasetEntry, Registry, get_registry

MIN_DEPTH, MAX_DEPTH = 0.1, 200.0
# carizon carries no LiDAR GT; sparse triangulation points serve as calibration
# references instead. ?calib=seq|feye selects the scheme; default falls back
# through depth_gt_path -> calib_seq_path -> calib_mono_feye_path.
# carizon_ccar HAS real LiDAR GT (depth_gt_path): a selected calib scheme that
# the frame lacks falls through to the default chain instead of returning None.
CALIB_FIELDS = {"seq": "calib_seq_path", "feye": "calib_mono_feye_path"}

# Raw outputs saved as inverse depth (keep in sync with eval_mmde.py IS_DISP).
DISP_MODELS = {"dav2_rel", "zipdepth", "gemdepth"}

# Split-root triangulation dirs double as sparse-control sources; the npz path
# for a given frame comes from its frames.jsonl field.
_CONTROL_SOURCE_FIELDS = {
    "calib_seq_triangulation": "calib_seq_path",
    "calib_mono_feye_triangulation": "calib_mono_feye_path",
}


# ---------------------------------------------------------------------------
# small numeric helpers (local replacements for the Manydepth2 sys.path hack)
# ---------------------------------------------------------------------------

def fit_affine(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Least-squares a*pred + b ~= gt."""
    A = np.stack([pred.astype(np.float64), np.ones(len(pred), np.float64)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, gt.astype(np.float64), rcond=None)
    return float(a), float(b)


def sample_pred(pred: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample pred at uv (N,2) pixel coords; out-of-bounds -> NaN.

    Pure numpy on purpose: cv2.remap crashes on 4K maps.
    """
    uv = np.asarray(uv, dtype=np.float64)
    u, v = uv[:, 0], uv[:, 1]
    h, w = pred.shape[:2]
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    dx = u - x0
    dy = v - y0
    # need the full 2x2 stencil in-bounds; boundary pixels are NaN
    valid = (x0 >= 0) & (y0 >= 0) & (x1 <= w - 1) & (y1 <= h - 1)
    out = np.full(len(uv), np.nan, np.float64)
    if valid.any():
        p = pred.astype(np.float64)
        m = valid
        wa = (1 - dx[m]) * (1 - dy[m])
        wb = dx[m] * (1 - dy[m])
        wc = (1 - dx[m]) * dy[m]
        wd = dx[m] * dy[m]
        out[m] = (p[y0[m], x0[m]] * wa + p[y0[m], x1[m]] * wb
                  + p[y1[m], x0[m]] * wc + p[y1[m], x1[m]] * wd)
    return out


def _quat_wxyz(R: np.ndarray) -> list[float]:
    """Rotation matrix -> [w,x,y,z], branch-stable standard extraction."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return [0.25 / s, (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                0.25 * s, (R[1, 2] + R[2, 1]) / s]
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s, 0.25 * s]


# ---------------------------------------------------------------------------
# caches
# ---------------------------------------------------------------------------

class LruCache:
    """Thread-safe in-memory LRU (old server CACHE equivalent)."""

    def __init__(self, max_items: int = 48):
        self._d: OrderedDict = OrderedDict()
        self._max = max_items
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, val) -> None:
        with self._lock:
            self._d[key] = val
            self._d.move_to_end(key)
            while len(self._d) > self._max:
                self._d.popitem(last=False)


_disk_cache: DiskCache | None = None
_disk_cache_lock = threading.Lock()
RGB_CACHE = LruCache(get_settings().rgb_cache_items)  # resized on configure()
FRAMES_CACHE: dict = {}
FRAMES_LOCK = threading.Lock()


def get_disk_cache() -> DiskCache:
    global _disk_cache
    with _disk_cache_lock:
        if _disk_cache is None:
            s = get_settings()
            _disk_cache = DiskCache(s.cache_dir, s.disk_cache_max_bytes)
    return _disk_cache


def configure_caches() -> None:
    """Called on (re)configure: rebuild rgb cache capacity."""
    global RGB_CACHE
    RGB_CACHE = LruCache(get_settings().rgb_cache_items)


def _mtime(path: Path) -> int:
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return -1


# ---------------------------------------------------------------------------
# frames / rgb / gt
# ---------------------------------------------------------------------------

def load_frames(entry: DatasetEntry, split: str) -> list[dict]:
    key = (entry.name, split)
    jsonl = entry.test_root / split / "frames.jsonl"
    mtime = _mtime(jsonl)
    with FRAMES_LOCK:
        cached = FRAMES_CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
    split_dir = entry.test_root / split
    frames = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frame = json.loads(line)
                frame["_split_dir"] = str(split_dir)
                frame["_test_root"] = str(entry.test_root)
                frames.append(frame)
    with FRAMES_LOCK:
        FRAMES_CACHE[key] = (mtime, frames)
    return frames


def frame_or_none(entry: DatasetEntry, split: str, idx: int):
    frames = load_frames(entry, split)
    if 0 <= idx < len(frames):
        return frames[idx]
    return None


def resolve_frame_path(frame: dict, value: str) -> str:
    """Resolve a path stored in frames.jsonl on a migrated deployment.

    frames.jsonl records absolute paths from the machine that built the test
    set.  When the bundle is unpacked elsewhere those paths may not exist;
    fall back in order:
      1. the recorded path itself;
      2. if it points inside an ``mmde_test/`` tree, the same relative tail
         under this deployment's test_root;
      3. ``<split_dir>/<basename>`` and ``<split_dir>/images/<basename>``
         (images bundled next to frames.jsonl).
    The original path is returned when nothing resolves so callers raise the
    familiar error.
    """
    p = Path(value)
    if p.is_file():
        return value
    split_dir = frame.get("_split_dir")
    parts = p.parts
    if "mmde_test" in parts:
        tail = parts[parts.index("mmde_test") + 1:]
        cand = Path(frame.get("_test_root", "")).joinpath(*tail)
        if cand.is_file():
            return str(cand)
    if split_dir:
        for cand in (Path(split_dir) / p.name,
                     Path(split_dir) / "images" / p.name):
            if cand.is_file():
                return str(cand)
    return value


def load_rgb(frame: dict) -> np.ndarray:
    image_path = resolve_frame_path(frame, frame["image_path"])
    rgb = RGB_CACHE.get(("rgb", image_path))
    if rgb is None:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        RGB_CACHE.put(("rgb", image_path), rgb)
    return rgb


def gt_path_for(frame: dict, calib: str | None) -> str | None:
    if calib in CALIB_FIELDS and frame.get(CALIB_FIELDS[calib]):
        return resolve_frame_path(frame, frame[CALIB_FIELDS[calib]])
    for key in ("depth_gt_path", "calib_seq_path", "calib_mono_feye_path"):
        if frame.get(key):
            return resolve_frame_path(frame, frame[key])
    return None


def load_gt(frame: dict, calib: str | None = None):
    path = gt_path_for(frame, calib)
    gt = RGB_CACHE.get(("gt", path))
    if gt is None:
        data = np.load(path)
        gt = (data["uv"].astype(np.float32), data["depth"].astype(np.float32))
        RGB_CACHE.put(("gt", path), gt)
    return gt


# ---------------------------------------------------------------------------
# binary encoders (layouts identical to pc_viewer_server.py)
# ---------------------------------------------------------------------------

def _affine_fit(depth: np.ndarray, frame: dict, calib: str | None) -> tuple[float, float]:
    """Per-frame affine fit of depth to sparse GT; identity when no GT."""
    if gt_path_for(frame, calib):
        uv, gt = load_gt(frame, calib)
        pv = sample_pred(depth, uv)
        valid = (gt > MIN_DEPTH) & (gt < MAX_DEPTH) & np.isfinite(pv) \
            & (pv > 1e-6) & (pv < 1000.0)
        if valid.sum() >= 10:
            return fit_affine(pv[valid], gt[valid])
    return 1.0, 0.0


def _depth_for_model(entry: DatasetEntry, split: str, idx: int, model: str,
                     rgb: np.ndarray) -> np.ndarray:
    pred_path = entry.pred_root / split / model / f"{idx:06d}.npy"
    pred = np.load(pred_path).astype(np.float32)
    h, w = rgb.shape[:2]
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    # Same convention as eval_mmde.py: raw disparity maps are converted to
    # depth first, then the per-frame affine fit happens in depth space.
    # *_calib_* outputs are already metric depth and must not be inverted.
    if model in DISP_MODELS:
        return (1.0 / np.maximum(pred, 1e-6)).astype(np.float32)
    return pred


def encode_points(entry: DatasetEntry, split: str, idx: int, model: str,
                  calib: str | None = None) -> bytes:
    frame = frame_or_none(entry, split, idx)
    pred_path = entry.pred_root / split / model / f"{idx:06d}.npy"
    if frame is None or not pred_path.exists():
        raise FileNotFoundError(pred_path)

    key = ("pts", entry.name, split, idx, model, get_settings().stride, calib,
           _mtime(resolve_frame_path(frame, frame["image_path"])), _mtime(pred_path))
    hit = get_disk_cache().get(key)
    if hit is not None:
        return hit

    rgb = load_rgb(frame)
    depth = _depth_for_model(entry, split, idx, model, rgb)
    h, w = rgb.shape[:2]
    # carizon test_sequence carries no GT/calib points: serve raw depth with
    # an identity "alignment" instead of failing the whole dual view
    a, b = _affine_fit(depth, frame, calib)
    aligned = (a * depth + b).astype(np.float32)

    step = get_settings().stride
    rows = np.arange(0, h, step)
    cols = np.arange(0, w, step)
    sub = np.ix_(rows, cols)
    d_raw = depth[sub].astype(np.float32).reshape(-1)
    d_al = aligned[sub].astype(np.float32).reshape(-1)
    rgb_sub = rgb[sub].astype(np.uint8).reshape(-1, 3)

    buf = io.BytesIO()
    buf.write(np.array([len(rows), len(cols), step], dtype="<u4").tobytes())
    buf.write(d_raw.tobytes())
    buf.write(d_al.tobytes())
    buf.write(rgb_sub.tobytes())
    out = buf.getvalue()
    get_disk_cache().put(key, out)
    return out


def encode_gt(entry: DatasetEntry, split: str, idx: int,
              calib: str | None = None) -> bytes:
    frame = frame_or_none(entry, split, idx)
    if frame is None or not gt_path_for(frame, calib):
        uv = np.zeros((0, 2), np.float32)
        d = np.zeros((0,), np.float32)
        gt_mtime = -1
    else:
        path = gt_path_for(frame, calib)
        gt_mtime = _mtime(path)
        uv, d = load_gt(frame, calib)

    key = ("gtbin", entry.name, split, idx, calib, gt_mtime)
    hit = get_disk_cache().get(key)
    if hit is not None:
        return hit
    buf = io.BytesIO()
    buf.write(np.array([uv.shape[0]], dtype="<u4").tobytes())
    buf.write(uv[:, 0].astype("<f4").tobytes())
    buf.write(uv[:, 1].astype("<f4").tobytes())
    buf.write(d.astype("<f4").tobytes())
    out = buf.getvalue()
    get_disk_cache().put(key, out)
    return out


# ---------------------------------------------------------------------------
# sparse controls
# ---------------------------------------------------------------------------

def _control_npz(entry: DatasetEntry, split: str, idx: int, source: str,
                 frame: dict) -> Path:
    p = entry.test_root / split / "sparse_controls" / source / f"{idx:06d}.npz"
    if p.is_file():
        return p
    field = _CONTROL_SOURCE_FIELDS.get(source)
    if field:
        fp = (frame or {}).get(field)
        if fp and Path(fp).is_file():
            return Path(fp)
    raise FileNotFoundError(f"{entry.name}/{split}/{idx}/{source}")


def load_controls(entry: DatasetEntry, split: str, idx: int, source: str):
    """Sparse control points (uv, depth, weight) from a control source.

    Triangulation npz (carizon calib) carry no weight column -> ones.
    """
    frame = frame_or_none(entry, split, idx)
    if frame is None:
        raise FileNotFoundError(idx)
    data = np.load(_control_npz(entry, split, idx, source, frame))
    uv = data["uv"].astype(np.float32)
    depth = data["depth"].astype(np.float32)
    weight = data["weight"].astype(np.float32) if "weight" in data \
        else np.ones(len(depth), np.float32)
    return uv, depth, weight


def encode_controls(entry: DatasetEntry, split: str, idx: int, source: str,
                    model: str | None = None,
                    calib: str | None = None) -> bytes:
    """Binary sparse-control payload for diagnostics.

    u32 n | f32 u[n] | f32 v[n] | f32 d_ctrl[n] | f32 w[n]
          | f32 d_pred_aligned[n] (NaN where the model has no prediction)
          | f32 err_gt[n] = |log(d_ctrl/gt)| at the nearest GT point within
            3 px (NaN where no GT matches)

    d_pred_aligned uses the same convention as /api/points: disparity models
    are inverted, then the per-frame affine fit to sparse GT (a, b) is applied,
    so |log(d_ctrl/d_pred_aligned)| measures control-vs-model inconsistency on
    the same scale the viewer displays.
    """
    frame = frame_or_none(entry, split, idx)
    if frame is None:
        raise FileNotFoundError(idx)
    ctrl_path = _control_npz(entry, split, idx, source, frame)
    uv, d_ctrl, w = load_controls(entry, split, idx, source)
    n = len(d_ctrl)

    pred_path = None
    if model:
        candidate = entry.pred_root / split / model / f"{idx:06d}.npy"
        if candidate.is_file():
            pred_path = candidate

    key = ("ctrl", entry.name, split, idx, source, model, calib,
           _mtime(ctrl_path), _mtime(resolve_frame_path(frame, frame["image_path"])),
           _mtime(pred_path) if pred_path else -1)
    hit = get_disk_cache().get(key)
    if hit is not None:
        return hit

    d_pred = np.full(n, np.nan, np.float32)
    err_gt = np.full(n, np.nan, np.float32)

    if pred_path is not None:
        rgb = load_rgb(frame)
        depth = _depth_for_model(entry, split, idx, model, rgb)
        a, b = _affine_fit(depth, frame, calib)
        d_pred = sample_pred((a * depth + b).astype(np.float32), uv) \
            .astype(np.float32)

    if gt_path_for(frame, calib):
        guv, gt = load_gt(frame, calib)
        from scipy.spatial import cKDTree
        dist, ii = cKDTree(guv).query(uv, k=1)
        m = dist < 3.0
        with np.errstate(divide="ignore", invalid="ignore"):
            err_gt[m] = np.abs(np.log(d_ctrl[m] / gt[ii[m]]))

    buf = io.BytesIO()
    buf.write(np.array([n], dtype="<u4").tobytes())
    for arr in (uv[:, 0], uv[:, 1], d_ctrl, w, d_pred, err_gt):
        buf.write(arr.astype("<f4").tobytes())
    out = buf.getvalue()
    get_disk_cache().put(key, out)
    return out


# ---------------------------------------------------------------------------
# scene index enrichment
# ---------------------------------------------------------------------------

def enrich_scene_index(data: bytes, entry: DatasetEntry, split: str) -> bytes:
    """Attach cam_quat / cam_t from frames.jsonl when the fused index lacks
    them (pre-fly-through fuse runs) -- fly metadata without re-fusing."""
    idx = json.loads(data)
    if "cam_quat" in idx and "cam_t" in idx:
        return data
    try:
        frames = load_frames(entry, split)
    except OSError:
        return data
    if len(frames) != idx.get("n_frames"):
        return data
    quats, ts, t0 = [], [], None
    for f in frames:
        quats.append(_quat_wxyz(np.asarray(f["T_world_camera"], np.float64)[:3, :3]))
        t = float(f["timestamp"])
        t0 = t if t0 is None else t0
        ts.append(round(t - t0, 3))
    idx["cam_quat"] = [[round(v, 5) for v in q] for q in quats]
    idx["cam_t"] = ts
    return json.dumps(idx).encode("utf-8")
