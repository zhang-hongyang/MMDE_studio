#!/usr/bin/env python3
"""Build 16x16 DTM label grids aligned to LFM patch tokens for each tile.

Reads the input256 tile manifest, crops the region DTM at each tile window,
downsamples to 16x16 (patch grid of a 256px input with patch size 16), and
saves float32 npy plus a validity mask.

Run with any env that has rasterio + numpy (e.g. lunarecon conda env).
"""
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

DATA = Path("/mnt/d/nac/official_rdr/expanded")
ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "outputs/marigold_v2_input256/tiles_manifest.csv"
OUT = ROOT / "outputs/lfm_probe/labels"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    rows = list(csv.DictReader(MANIFEST.open()))
    by_region = {}
    for row in rows:
        by_region.setdefault(row["region"], []).append(row)
    for region, rrows in sorted(by_region.items()):
        dtm_path = DATA / region / f"NAC_DTM_{region}.TIF"
        with rasterio.open(dtm_path) as ds:
            for row in rrows:
                out = OUT / f"{region}_{row['file'].replace('.png', '')}_label.npy"
                if out.exists():
                    continue
                x, y, w, h = (int(row[k]) for k in ("x", "y", "width", "height"))
                a = ds.read(1, window=Window(x, y, w, h), masked=True)
                gt = np.ma.getmaskarray(a) | ~np.isfinite(a.data)
                valid = ~gt
                a = np.where(valid, a.data, np.nan)
                # pad edge tiles (w/h may be < 512) by nan-padding to 512
                if w < 512 or h < 512:
                    pad = np.full((512, 512), np.nan, np.float32)
                    pad[:h, :w] = a
                    a = pad
                    v = np.zeros((512, 512), bool)
                    v[:h, :w] = valid
                    valid = v
                # 512 -> 16x16 patch means
                lbl = np.nanmean(a.reshape(16, 32, 16, 32).transpose(0, 2, 1, 3), axis=(2, 3))
                msk = valid.reshape(16, 32, 16, 32).transpose(0, 2, 1, 3).mean(axis=(2, 3))
                np.save(out, np.stack([lbl, (msk >= 0.9).astype(np.float32)]))
        print(region, "done", flush=True)
    print("labels:", len(list(OUT.glob("*_label.npy"))))


if __name__ == "__main__":
    main()
