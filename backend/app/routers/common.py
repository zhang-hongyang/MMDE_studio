"""Shared helpers for API routers: name validation and registry lookups."""
from __future__ import annotations

import re

from fastapi import HTTPException

from ..registry import (DatasetEntry, SplitInfo, UnknownDatasetError,
                        get_registry)

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _bad_name(s: str) -> bool:
    # NAME_RE alone matches "." / ".."; guard against both.
    return not NAME_RE.match(s) or s in (".", "..")


def check_name(s: str, what: str = "name") -> str:
    if _bad_name(s):
        raise HTTPException(status_code=404, detail=f"invalid {what}")
    return s


def get_dataset(name: str) -> DatasetEntry:
    check_name(name, "dataset")
    try:
        return get_registry().get_dataset(name)
    except UnknownDatasetError:
        raise HTTPException(status_code=404, detail="unknown dataset") from None


def get_split(entry: DatasetEntry, split: str) -> SplitInfo:
    check_name(split, "split")
    from ..registry import UnknownSplitError
    try:
        return get_registry().split_info(entry, split)
    except UnknownSplitError:
        raise HTTPException(status_code=404, detail="unknown split")


def parse_calib(calib: str | None) -> str | None:
    from ..data_access import CALIB_FIELDS
    return calib if calib in CALIB_FIELDS else None
