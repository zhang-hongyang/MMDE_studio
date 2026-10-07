"""Fused multi-frame scene endpoints (built by fuse_scene.py)."""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from .. import data_access as da
from ..registry import get_registry
from .common import check_name, get_dataset, get_split

router = APIRouter()

SCENE_BLOB_RE = re.compile(
    r"^(?:overview|overview_s\d+|chunk_\d{3}|chunk_s\d+_\d{3})\.bin$")


@router.get("/api/scenes")
def list_scenes():
    # every dataset/split combo holding >=1 fused scene; the viewer uses it
    # to auto-switch when the current combo has none
    reg = get_registry()
    out = []
    for name in reg.names():
        entry = reg.get_dataset(name)
        for split, info in reg.splits(entry).items():
            if info.scene_models:
                out.append({"dataset": name, "split": split,
                            "models": info.scene_models})
    return out


@router.get("/api/scene_models/{ds}/{split}")
def list_scene_models(ds: str, split: str):
    entry = get_dataset(ds)
    info = get_split(entry, split)
    return info.scene_models


@router.get("/api/scene/{ds}/{split}/{model}/index.json")
def scene_index(ds: str, split: str, model: str):
    entry = get_dataset(ds)
    get_split(entry, split)
    check_name(model, "model")
    path = entry.scene_root / split / model / "index.json"
    key = ("sceneidx", entry.name, split, model, da._mtime(path))
    data = da.get_disk_cache().get(key)
    if data is None:
        try:
            raw = path.read_bytes()
        except OSError:
            raise HTTPException(status_code=404, detail="scene not found")
        data = da.enrich_scene_index(raw, entry, split)
        da.get_disk_cache().put(key, data)
    # no browser cache: the index can gain fields (enrichment) or be re-fused
    # under the same URL; blobs carry ?v= busting instead
    return Response(data, media_type="application/json",
                    headers={"Cache-Control": "no-cache"})


@router.get("/api/scene/{ds}/{split}/{model}/blob/{name}")
def scene_blob(ds: str, split: str, model: str, name: str):
    entry = get_dataset(ds)
    get_split(entry, split)
    check_name(model, "model")
    if not SCENE_BLOB_RE.match(name):
        raise HTTPException(status_code=404, detail="invalid blob name")
    path = entry.scene_root / split / model / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="blob not found")
    return FileResponse(path, media_type="application/octet-stream",
                        headers={"Cache-Control": "public, max-age=3600"})
