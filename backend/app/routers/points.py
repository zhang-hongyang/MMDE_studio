"""Model list, GT / points / controls binary endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from .. import data_access as da
from .common import check_name, get_dataset, get_split, parse_calib

router = APIRouter()

OCTET = "application/octet-stream"


def _octet(data: bytes) -> Response:
    return Response(data, media_type=OCTET,
                    headers={"Cache-Control": "public, max-age=3600"})


@router.get("/api/models/{ds}/{split}")
def list_models(ds: str, split: str):
    entry = get_dataset(ds)
    info = get_split(entry, split)
    return info.models


@router.get("/api/gt/{ds}/{split}/{idx}")
def gt_binary(ds: str, split: str, idx: int, calib: str | None = None):
    entry = get_dataset(ds)
    get_split(entry, split)
    try:
        return _octet(da.encode_gt(entry, split, idx, parse_calib(calib)))
    except (OSError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="not found") from None


@router.get("/api/points/{ds}/{split}/{idx}/{model}")
def points_binary(ds: str, split: str, idx: int, model: str,
                  calib: str | None = None):
    entry = get_dataset(ds)
    get_split(entry, split)
    check_name(model, "model")
    try:
        return _octet(da.encode_points(entry, split, idx, model,
                                       parse_calib(calib)))
    except (OSError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="not found") from None


@router.get("/api/control_sources")
def control_sources(dataset: str = "kitti", split: str = "test_sequence"):
    entry = get_dataset(dataset)
    info = get_split(entry, split)
    return info.control_sources


@router.get("/api/controls/{ds}/{split}/{idx}/{source}")
def controls_binary(ds: str, split: str, idx: int, source: str,
                    model: str | None = None, calib: str | None = None):
    entry = get_dataset(ds)
    get_split(entry, split)
    check_name(source, "source")
    if model:
        check_name(model, "model")
    try:
        return _octet(da.encode_controls(entry, split, idx, source, model,
                                         parse_calib(calib)))
    except (OSError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="not found") from None
