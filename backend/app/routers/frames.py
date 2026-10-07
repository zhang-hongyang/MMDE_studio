"""Frame listing, per-frame meta and RGB serving (original + thumbnails)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response

from .. import data_access as da
from .common import get_dataset, get_split

router = APIRouter()

CACHE_1H = {"Cache-Control": "public, max-age=3600"}


def _frame_or_404(entry, split, idx):
    frame = da.frame_or_none(entry, split, idx)
    if frame is None:
        raise HTTPException(status_code=404, detail="frame not found")
    return frame


@router.get("/api/frames")
def list_frames(dataset: str = "kitti", split: str = "test_sequence",
                camera: str | None = None):
    entry = get_dataset(dataset)
    get_split(entry, split)  # 404 on unknown split
    frames = da.load_frames(entry, split)
    out = []
    for i, f in enumerate(frames):
        c = f.get("camera")
        if camera and c != camera:
            continue
        item = {"idx": f.get("_index", i), "seq_id": f["seq_id"],
                "frame_id": f["frame_id"]}
        if c:
            item["camera"] = c
        out.append(item)
    return out


@router.get("/api/meta/{ds}/{split}/{idx}")
def frame_meta(ds: str, split: str, idx: int):
    entry = get_dataset(ds)
    info = get_split(entry, split)
    frame = _frame_or_404(entry, split, idx)
    h = w = None
    try:
        rgb = da.load_rgb(frame)
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
    except (OSError, FileNotFoundError):
        # image unreadable: infer from the first available depth prediction
        for model in info.models:
            p = entry.pred_root / split / model / f"{idx:06d}.npy"
            if p.is_file():
                h, w = np.load(p, mmap_mode="r").shape
                break
    if h is None:
        raise HTTPException(status_code=404, detail="image unavailable")
    return {
        "K": frame["K"], "w": w, "h": h,
        "seq_id": frame["seq_id"], "frame_id": frame["frame_id"],
        # camera->world pose; lets the viewer register two frames of a
        # sequence into one coordinate system for temporal diffing
        "T": frame.get("T_world_camera"),
    }


@router.get("/api/rgb/{ds}/{split}/{idx}")
def frame_rgb(ds: str, split: str, idx: int,
              w: int = Query(0, description="thumbnail width; 0 = original")):
    entry = get_dataset(ds)
    get_split(entry, split)
    frame = _frame_or_404(entry, split, idx)

    img_path = Path(da.resolve_frame_path(frame, frame["image_path"]))
    try:
        w_max = int(w)
    except (TypeError, ValueError):
        w_max = 0

    if w_max > 0:
        # Thumbnail for fly-through playback: full-res frames are ~1.5 MB
        # each (4K for carizon); downscale server-side so streaming one per
        # frame stays cheap (replays ride on the browser/disk cache). The
        # long edge is resized to the target width BEFORE encoding; the old
        # cv2.remap-based sampler crashed on these 4K images.
        key = ("rgbsm", str(img_path), da._mtime(img_path), w_max)
        hit = da.get_disk_cache().get(key)
        if hit is None:
            rgb = da.load_rgb(frame)
            h, wd = rgb.shape[:2]
            if w_max < wd:
                rgb = cv2.resize(rgb, (w_max, max(1, round(h * w_max / wd))),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise HTTPException(status_code=404, detail="encode failed")
            hit = buf.tobytes()
            da.get_disk_cache().put(key, hit)
        return Response(hit, media_type="image/jpeg", headers=CACHE_1H)

    key = ("rgbfull", str(img_path), da._mtime(img_path))
    hit = da.get_disk_cache().get(key)
    if hit is None:
        hit = img_path.read_bytes()
        da.get_disk_cache().put(key, hit)
    return Response(hit, media_type="image/jpeg", headers=CACHE_1H)
