"""Mark skipped low-coverage windows NoData and rescore saved predictions."""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


DATA = Path("/mnt/d/nac/official_rdr/expanded")
BASE = Path(__file__).parent / "outputs"


def windows(dataset):
    for y in range(0, dataset.height, 512):
        for x in range(0, dataset.width, 512):
            yield Window(x, y, min(512, dataset.width - x),
                         min(512, dataset.height - y))


def valid_arrays(dtm, prediction, window):
    gt = dtm.read(1, window=window, masked=True)
    pred = prediction.read(1, window=window)
    valid = (~np.ma.getmaskarray(gt) & np.isfinite(gt.data) &
             np.isfinite(pred) & (pred != -9999))
    return gt.data, pred, valid


def repair_and_score(region, folder):
    dtm_path = DATA / region / f"NAC_DTM_{region}.TIF"
    pred_path = folder / f"{region}_prediction.tif"
    skipped_windows = skipped_valid = 0
    with rasterio.open(dtm_path) as dtm, rasterio.open(pred_path, "r+") as prediction:
        for window in windows(dtm):
            gt = dtm.read(1, window=window, masked=True)
            valid = ~np.ma.getmaskarray(gt) & np.isfinite(gt.data)
            if valid.mean() < 0.1:
                skipped_windows += 1
                skipped_valid += int(valid.sum())
                prediction.write(np.full(gt.shape, -9999, dtype=np.float32),
                                 1, window=window)
        n = sx = sy = sxx = sxy = 0.0
        for window in windows(dtm):
            gt, pred, valid = valid_arrays(dtm, prediction, window)
            if not valid.any():
                continue
            x = pred[valid].astype(np.float64)
            y = gt[valid].astype(np.float64)
            n += len(x); sx += x.sum(); sy += y.sum()
            sxx += x @ x; sxy += x @ y
        denominator = sxx - sx * sx / n
        scale = ((sxy - sx * sy / n) / denominator
                 if denominator > 0 else 0.0)
        offset = (sy - scale * sx) / n
        mean = sy / n
        ae = se = constant_ae = constant_se = 0.0
        for window in windows(dtm):
            gt, pred, valid = valid_arrays(dtm, prediction, window)
            if not valid.any():
                continue
            x = pred[valid].astype(np.float64)
            y = gt[valid].astype(np.float64)
            error = scale * x + offset - y
            constant_error = mean - y
            ae += np.abs(error).sum(); se += error @ error
            constant_ae += np.abs(constant_error).sum()
            constant_se += constant_error @ constant_error
    metric = {"region": region, "pixels": int(n),
              "oracle_affine_scale": scale, "oracle_affine_offset": offset,
              "oracle_mae_m": ae / n, "oracle_rmse_m": np.sqrt(se / n)}
    baseline = {"region": region, "pixels": int(n),
                "oracle_constant_mae_m": constant_ae / n,
                "oracle_constant_rmse_m": np.sqrt(constant_se / n)}
    return metric, baseline, skipped_windows, skipped_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", choices=["dav2_small", "dav2_small_input256",
                                          "marigold_v1_1_input256"])
    args = parser.parse_args()
    folder = BASE / args.folder
    metrics_path = folder / "region_metrics.csv"
    with metrics_path.open(newline="") as file:
        original = list(csv.DictReader(file))
    if len(original) != 20 or any(row["complete"] != "True" for row in original):
        raise ValueError("Only repair a completed 20-region evaluation")
    backup = folder / "region_metrics_before_mask_fix.csv"
    if not backup.exists():
        shutil.copy2(metrics_path, backup)
    baselines = []
    audit = []
    for row in original:
        metric, baseline, skipped_windows, skipped_valid = repair_and_score(
            row["region"], folder)
        row.update(metric)
        baselines.append(baseline)
        audit.append({"region": row["region"],
                      "skipped_windows": skipped_windows,
                      "valid_pixels_in_skipped_windows": skipped_valid})
        print(audit[-1], flush=True)
    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=original[0].keys())
        writer.writeheader(); writer.writerows(original)
    with (folder / "constant_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=baselines[0].keys())
        writer.writeheader(); writer.writerows(baselines)
    with (folder / "boundary_mask_audit.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=audit[0].keys())
        writer.writeheader(); writer.writerows(audit)
    print("Total skipped valid pixels:",
          sum(row["valid_pixels_in_skipped_windows"] for row in audit))


if __name__ == "__main__":
    main()
