"""Precompute frozen DINOv3-S multiscale features for all materialized tiles.

Features are fold-independent (the backbone is frozen and never sees labels),
so they are computed once for all 20 regions and reused by every fold.
Stored as float16 .npy per tile, 4 stages x 384 ch at 32x32 for 512px input.
"""

import time
from pathlib import Path

import numpy as np
import torch
import timm

DATA = Path("/mnt/d/nac/supervised_v1")
OUT = DATA / "dinov3_s"
TILE = 512
STAGE_LAYERS = (2, 5, 8, 11)  # 1-indexed-ish block indices for 4 stages


def main():
    model = timm.create_model("vit_small_patch16_dinov3", pretrained=True,
                              num_classes=0).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    OUT.mkdir(parents=True, exist_ok=True)

    tiles = sorted(DATA.glob("*/tiles/*_img.npy"))
    print(f"{len(tiles)} tiles")
    t0 = time.time()
    done = 0
    B = 16
    batch_imgs, batch_keys = [], []

    def flush():
        nonlocal batch_imgs, batch_keys, done
        if not batch_imgs:
            return
        # Edge tiles are partial; reflect-pad to the full 512 square (the
        # valid-pixel mask, not the padding, governs training/eval).
        batch_imgs = [np.pad(a, ((0, TILE - a.shape[0]), (0, TILE - a.shape[1])),
                             mode="edge") for a in batch_imgs]
        x = torch.from_numpy(np.stack(batch_imgs)).cuda().float() / 255.0
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        with torch.inference_mode():
            _, feats = model.forward_intermediates(
                x, indices=list(STAGE_LAYERS), return_prefix_tokens=False,
                output_fmt="NCHW")
        for j, key in enumerate(batch_keys):
            arr = np.stack([f[j].half().cpu().numpy() for f in feats])
            np.save(OUT / f"{key}_dino.npy", arr)
        done += len(batch_keys)
        batch_imgs, batch_keys = [], []

    for path in tiles:
        key = path.name.removesuffix("_img.npy")
        region = path.parent.parent.name
        out_path = OUT / f"{region}_{key}_dino.npy"
        if out_path.exists():
            continue
        batch_imgs.append(np.load(path))
        batch_keys.append(f"{region}_{key}")
        if len(batch_imgs) == B:
            flush()
            if done % 400 < B:
                el = time.time() - t0
                print(f"{done} tiles, {el:.0f}s, eta {el/max(done,1)*(len(tiles)-done):.0f}s",
                      flush=True)
    flush()
    # cleanup stale progress markers if any
    print(f"done: {done} new tiles in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
