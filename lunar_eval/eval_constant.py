"""Oracle constant-elevation sanity baseline on DAV2's evaluated pixels."""

import csv
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path("/mnt/d/nac/official_rdr/expanded")
OUT = Path(__file__).parent / "outputs/dav2_small"


def main():
    with (OUT / "region_metrics.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    results = []
    output_path = OUT / "constant_metrics.csv"
    if output_path.exists():
        with output_path.open(newline="") as file:
            results = list(csv.DictReader(file))
    completed = {row["region"] for row in results}
    for row in rows:
        region = row["region"]
        if row["complete"] != "True" or region in completed:
            continue
        gt_path = ROOT / region / f"NAC_DTM_{region}.TIF"
        pred_path = OUT / f"{region}_prediction.tif"
        with rasterio.open(gt_path) as gt, rasterio.open(pred_path) as pred:
            count = total = 0.0
            for _, window in gt.block_windows(1):
                reference = gt.read(1, window=window, masked=True)
                prediction = pred.read(1, window=window)
                valid = (~np.ma.getmaskarray(reference) &
                         np.isfinite(reference.data) & (prediction != -9999))
                count += valid.sum()
                total += reference.data[valid].astype(np.float64).sum()
            mean = total / count
            ae = se = 0.0
            for _, window in gt.block_windows(1):
                reference = gt.read(1, window=window, masked=True)
                prediction = pred.read(1, window=window)
                valid = (~np.ma.getmaskarray(reference) &
                         np.isfinite(reference.data) & (prediction != -9999))
                error = reference.data[valid].astype(np.float64) - mean
                ae += np.abs(error).sum()
                se += error @ error
        result = {"region": region, "pixels": int(count),
                  "oracle_constant_mae_m": ae / count,
                  "oracle_constant_rmse_m": np.sqrt(se / count)}
        results.append(result)
        print(result, flush=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
