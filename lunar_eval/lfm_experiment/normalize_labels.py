#!/usr/bin/env python3
"""Normalize DTM labels to z-scores (global train stats) in-place.

DTM labels in raw meters (~-1000..+3000) make Adam take ~lr-sized steps, so the
head needs millions of steps to reach the output offset. Z-scored labels are
O(1) and train in reasonable time.

Rewrites data/nac_dtm/*_DTM.tif as (x-mean)/std, keeps nodata=-1, and writes
data/nac_dtm/label_stats.json {mean, std, nodata}. eval_finetune.py must
un-normalize predictions with the same stats.

Run AFTER prep_finetune_data.py. Idempotent: refuses to double-normalize.
"""
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/nac_dtm"
NODATA = -1.0
STATS = DATA / "label_stats.json"


def read_valid(p):
    with rasterio.open(p) as d:
        a = d.read(1, masked=True)
    return a.data[~np.ma.getmaskarray(a)]


def main():
    train_stems = [Path(l).stem for l in (DATA / "splits/train.txt").read_text().split()]
    # global stats over train tiles only (two passes: mean, then var)
    n, s, ss = 0, 0.0, 0.0
    for s_ in train_stems:
        v = read_valid(DATA / f"{s_}_DTM.tif")
        n += v.size; s += v.sum(); ss += (v * v).sum()
    mean = s / n
    std = np.sqrt(ss / n - mean**2)
    print(f"train tiles {len(train_stems)}  pixels {n/1e6:.1f}M  mean {mean:.2f} m  std {std:.2f} m")
    if STATS.exists():
        old = json.loads(STATS.read_text())
        if abs(old["std"] - 1.0) < 0.2:
            raise SystemExit(f"labels already normalized (stats={old}); aborting")
    for split in ("train", "val"):
        for s_ in [Path(l).stem for l in (DATA / f"splits/{split}.txt").read_text().split()]:
            p = DATA / f"{s_}_DTM.tif"
            with rasterio.open(p) as d:
                a = d.read(1, masked=True)
                prof = d.profile
            m = ~np.ma.getmaskarray(a)
            g = np.where(m, (a.data - mean) / std, NODATA).astype(np.float32)
            with rasterio.open(p, "w", **prof) as d:
                d.write(g, 1)
    STATS.write_text(json.dumps({"mean": float(mean), "std": float(std),
                                 "nodata": NODATA}, indent=2))
    print("rewrote labels as z-scores; wrote", STATS)


if __name__ == "__main__":
    main()
