"""Evaluate a trained DMBDepth checkpoint on one fold's test regions.

Writes per-region prediction GeoTIFFs on the identical 2 m DTM grid and a
region_metrics.csv with one oracle-affine fit per region (shape diagnostic,
same protocol as eval_dav2.py). Region means/scales are NOT available to the
model; the affine fit is applied only post-hoc for the diagnostic.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

from dmb_depth_model import DMBDepth
from train_dmb_depth import DATA, SPLITS, TileDataset

OUT_ROOT = Path(__file__).parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    splits = json.loads(SPLITS.read_text())
    fold = next(f for f in splits["folds"] if f["fold"] == args.fold)
    out_dir = OUT_ROOT / f"dmb_depth_fold{args.fold}"
    ckpt = torch.load(out_dir / "checkpoint_best.pt", map_location="cuda")
    model = DMBDepth().cuda()
    model.load_state_dict(ckpt["model"])
    print(f"loaded checkpoint from epoch {ckpt['epoch']} (val_mae={ckpt['val_mae']:.3f})")

    rows = []
    for region in fold["test"]:
        t0 = time.time()
        region_dir = DATA / region
        index = json.loads((region_dir / "tiles.json").read_text())
        georef = json.loads((region_dir / "georef.json").read_text())
        pred_full = np.full((georef["height"], georef["width"]), np.nan,
                            dtype=np.float32)

        # TileDataset sorts tile keys, matching the loader order; index by key.
        by_key = {e["key"]: e for e in index}

        ds = TileDataset([region])
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                             shuffle=False)
        model.eval()
        with torch.inference_mode():
            pos = 0
            for img, _, mask, dino, *_rest in loader:
                stages = [dino[:, i].contiguous().float().cuda() for i in range(4)]
                pred, _ = model(img.cuda(), stages)
                pred = pred[:, 0].cpu().numpy()
                m = mask[:, 0].numpy()
                for b in range(img.shape[0]):
                    key = ds.entries[pos][1]
                    entry = by_key[key]
                    tile = pred[b]
                    tm = m[b]
                    H = min(512, georef["height"] - entry["row"])
                    W = min(512, georef["width"] - entry["col"])
                    pred_full[entry["row"]:entry["row"] + H,
                              entry["col"]:entry["col"] + W] = np.where(
                        tm[:H, :W], tile[:H, :W], np.nan)
                    pos += 1

        # Reference DTM (identical grid) for scoring.
        dtm_path = Path("/mnt/d/nac/official_rdr/expanded") / region / f"NAC_DTM_{region}.TIF"
        with rasterio.open(dtm_path) as dtm:
            gt_full = dtm.read(1, masked=True)
            gt = gt_full.data.astype(np.float64)
            gt_mask = ~np.ma.getmaskarray(gt_full) & np.isfinite(gt)
        valid = gt_mask & np.isfinite(pred_full)
        pred = pred_full[valid].astype(np.float64)
        y = gt[valid]
        n = len(y)

        # Oracle-affine per region (shape diagnostic only).
        sx, sy = pred.sum(), y.sum()
        sxx, sxy = pred @ pred, pred @ y
        denom = sxx - sx * sx / n
        scale = (sxy - sx * sy / n) / denom if denom > 0 else 0.0
        offset = (sy - scale * sx) / n
        err = scale * pred + offset - y
        mae = np.abs(err).mean()
        rmse = np.sqrt((err @ err) / n)
        ss_res = (err @ err)
        ss_tot = ((y - y.mean()) @ (y - y.mean()))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        profile = {
            "driver": "GTiff", "height": georef["height"],
            "width": georef["width"], "crs": georef["crs"],
            "transform": rasterio.Affine(*georef["transform"]),
            "count": 1, "dtype": "float32", "nodata": -9999,
            "compress": "deflate", "predictor": 3,
        }
        pred_path = out_dir / f"{region}_prediction.tif"
        with rasterio.open(pred_path, "w", **profile) as dst:
            dst.write(np.where(np.isfinite(pred_full), pred_full, -9999), 1)

        rows.append({"region": region, "pixels": int(n), "coverage": n / gt_mask.sum(),
                     "oracle_affine_scale": scale, "oracle_affine_offset": offset,
                     "oracle_mae_m": mae, "oracle_rmse_m": rmse, "oracle_r2": r2})
        print(f"{region}: n={n:,} oracle MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    metrics_path = out_dir / "region_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
