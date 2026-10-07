"""Depth Anything V2 construction shared by PTC and MTD adapters."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

DEFAULT_MTD_ROOT = Path(
    "/home/ZhangHongyang/ResearchHub-workspaces/MMDE_studio/code/current/"
    "algorithms/mtd")
DEFAULT_CHECKPOINT = Path(
    "/home/ZhangHongyang/ResearchHub-scratch/models/MMDE_studio/"
    "depth-anything-v2/depth_anything_v2_vitl.pth")


def build_dav2(device: str):
    mtd_root = Path(os.environ.get("MMDE_MTD_ROOT", DEFAULT_MTD_ROOT))
    dav2_root = mtd_root / "third_party" / "Depth-Anything-V2"
    if not dav2_root.is_dir():
        raise FileNotFoundError(f"Depth Anything V2 source missing: {dav2_root}")
    sys.path.insert(0, str(dav2_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    checkpoint = Path(os.environ.get("DAV2_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DAV2 checkpoint missing: {checkpoint}")
    model = DepthAnythingV2(
        encoder="vitl", features=256,
        out_channels=[256, 512, 1024, 1024])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
