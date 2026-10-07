#!/usr/bin/env python3
"""Prepare NAC->DTM fine-tuning tiles in TerraTorch GenericMultimodalDataset layout.

Output (single flat dir, sample-prefix convention):
  data/nac_dtm/{PREFIX}_NAC.tif  float32 [0,1] 256x256 (from input256 PNGs)
  data/nac_dtm/{PREFIX}_DTM.tif  float32 meters 256x256, nodata=-1 (512->256 mean)
  data/nac_dtm/splits/{train,val}.txt  sample prefixes (fold0 of supervised_splits)

Run with lunarecon conda env (rasterio + PIL).
"""
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
DATA = Path("/mnt/d/nac/official_rdr/expanded")
PNG = ROOT / "outputs/marigold_v2_input256/inputs"
OUT = ROOT / "data/nac_dtm"
SPLITS = OUT / "splits"
NODATA = -1.0


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    splits = json.loads((ROOT / "outputs/supervised_splits.json").read_text())
    fold = splits["folds"][0]
    region_split = {**{r: "train" for r in fold["train"]},
                    **{r: "val" for r in fold["val"]}}
    print("fold0 train:", len(fold["train"]), "val:", fold["val"])

    prefixes = {"train": [], "val": []}
    rows = list(csv.DictReader((ROOT / "outputs/marigold_v2_input256/tiles_manifest.csv").open()))
    dt_handles = {}
    for r in rows:
        split = region_split.get(r["region"])
        if split is None:
            continue
        stem = r["file"].replace(".png", "")
        img_out = OUT / f"{stem}_NAC.tif"
        lbl_out = OUT / f"{stem}_DTM.tif"
        prefixes[split].append(stem)
        if img_out.exists() and lbl_out.exists():
            continue
        # image: existing 256px grayscale PNG -> float32 [0,1]
        arr = np.asarray(Image.open(PNG / r["file"]).convert("L"), np.float32) / 255.0
        h_img = {"driver": "GTiff", "height": 256, "width": 256,
                 "count": 1, "dtype": "float32"}
        with rasterio.open(img_out, "w", **h_img) as ds:
            ds.write(arr, 1)
        # label: DTM 512 window -> 256 mean, nodata=-1
        if r["region"] not in dt_handles:
            dt_handles[r["region"]] = rasterio.open(
                DATA / r["region"] / f"NAC_DTM_{r['region']}.TIF")
        ds = dt_handles[r["region"]]
        x, y, w, h = (int(r[k]) for k in ("x", "y", "width", "height"))
        a = ds.read(1, window=Window(x, y, w, h), masked=True)
        valid = ~np.ma.getmaskarray(a) & np.isfinite(a.data)
        a = np.where(valid, a.data, np.nan)
        if w < 512 or h < 512:
            pad = np.full((512, 512), np.nan, np.float32)
            pad[:h, :w] = a
            a = pad
        lbl = np.nanmean(a.reshape(256, 2, 256, 2).transpose(0, 2, 1, 3), axis=(2, 3))
        lbl = np.where(np.isfinite(lbl), lbl, NODATA).astype(np.float32)
        h_lbl = {"driver": "GTiff", "height": 256, "width": 256,
                 "count": 1, "dtype": "float32", "nodata": NODATA}
        with rasterio.open(lbl_out, "w", **h_lbl) as ds2:
            ds2.write(lbl, 1)
    for ds in dt_handles.values():
        ds.close()
    for split, names in prefixes.items():
        (SPLITS / f"{split}.txt").write_text("\n".join(sorted(names)) + "\n")
        print(split, len(names), "tiles")


if __name__ == "__main__":
    main()
