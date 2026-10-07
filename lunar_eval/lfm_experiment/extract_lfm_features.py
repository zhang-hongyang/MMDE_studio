#!/usr/bin/env python3
"""Extract frozen SOMA (NASA-IBM Lunar FM) NAC-encoder features per tile.

Input  : outputs/marigold_v2_input256/inputs/*.png (256px, same protocol as
         the DAV2/Marigold input256 benchmark track)
Output : outputs/lfm_probe/features/{tile}_f.npy  float16 (12, 256, 768)
         all 12 encoder block outputs, 256 tokens = 16x16 patches

Normalization follows the backbone config for the `nac` domain:
scaler std, mean 0.053995178164645, std 0.0374660347301751 on [0,1] pixels.

Run with .venv_lfm. GPU optional.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
INP = ROOT / "outputs/marigold_v2_input256/inputs"
OUT = ROOT / "outputs/lfm_probe/features"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT / "models/ni_lfm/backbone/checkpoint.pt"
CFG = ROOT / "models/ni_lfm/backbone/config.yaml"

NAC_MEAN = 0.053995178164645
NAC_STD = 0.0374660347301751
BATCH = 16


def main():
    from terratorch_integration.lunar_backbone import LunarBackbone

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bb = LunarBackbone(variant="base", modalities=["nac"], cfg=str(CFG),
                       checkpoint_path=str(CKPT))
    bb.eval().to(device)

    files = sorted(INP.glob("*.png"))
    print(f"{len(files)} tiles")
    done = 0
    for i in range(0, len(files), BATCH):
        batch = files[i:i + BATCH]
        tensors, outs = [], []
        for f in batch:
            out = OUT / f"{f.stem}_f.npy"
            if out.exists():
                continue
            arr = np.asarray(Image.open(f).convert("L"), np.float32) / 255.0
            t = torch.from_numpy((arr - NAC_MEAN) / NAC_STD)
            tensors.append(t)
            outs.append(out)
        if not tensors:
            continue
        xb = torch.stack(tensors).unsqueeze(1).to(device)  # (B,1,256,256)
        with torch.no_grad():
            feats = bb({"nac": xb})  # list of 12 x (B,256,768)
        stack = torch.stack([f.float().cpu() for f in feats]).numpy()
        for j, out in enumerate(outs):
            np.save(out, stack[:, j].astype(np.float16))
        done += len(outs)
        if done % 400 < BATCH:
            print(f"{done}/{len(files)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
