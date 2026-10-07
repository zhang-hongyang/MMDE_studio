"""Score all 20 regions on the same native 32 m reference grid."""

import csv
from pathlib import Path

import numpy as np
import rasterio


BASE = Path(__file__).parent / "outputs/coarse32"
METHODS = ("dav2", "marigold_v1", "marigold_v2")


def prediction_path(row, method):
    stem = Path(row["file"]).with_suffix(".npy")
    if method == "marigold_v2":
        return BASE / f"v2_batch{row['batch']}" / "images/predictions_npy" / stem
    return BASE / f"{method}_npy" / stem


def assemble(region, rows, method, shape):
    surface = np.full(shape, np.nan, dtype=np.float32)
    for row in rows:
        path = prediction_path(row, method)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {method} tile: {path}")
        tile = np.load(path)
        if tile.shape != (256, 256):
            raise ValueError(f"Unexpected prediction shape {tile.shape}: {path}")
        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["width"]), int(row["height"])
        surface[y:y+h, x:x+w] = tile[:h, :w]
    return surface


def score(z, prediction, valid):
    x = prediction[valid].astype(np.float64)
    y = z[valid].astype(np.float64)
    x0, y0 = x.mean(), y.mean()
    dx, dy = x - x0, y - y0
    denominator = dx @ dx
    scale = (dx @ dy) / denominator if denominator > 0 else 0.0
    offset = y0 - scale * x0
    error = scale * x + offset - y
    constant_error = y0 - y
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    constant_mae = np.mean(np.abs(constant_error))
    constant_rmse = np.sqrt(np.mean(constant_error**2))
    return {"pixels": int(valid.sum()), "oracle_affine_scale": scale,
            "oracle_affine_offset": offset, "oracle_mae_m": mae,
            "oracle_rmse_m": rmse, "oracle_r2": 1 - rmse**2 / constant_rmse**2,
            "oracle_constant_mae_m": constant_mae,
            "oracle_constant_rmse_m": constant_rmse}


def main():
    with (BASE / "manifest.csv").open(newline="") as file:
        manifest = list(csv.DictReader(file))
    by_region = {}
    for row in manifest:
        by_region.setdefault(row["region"], []).append(row)
    if len(by_region) != 20 or len(manifest) != 83:
        raise ValueError("Expected 20 regions and 83 coarse tiles")
    results = []
    for region, rows in sorted(by_region.items()):
        gt_path = BASE / "reference" / f"{region}_DTM_32m.tif"
        with rasterio.open(gt_path) as ds:
            z = ds.read(1)
            profile = ds.profile.copy()
        valid_gt = np.isfinite(z) & (z != -9999)
        surfaces = {method: assemble(region, rows, method, z.shape)
                    for method in METHODS}
        valid = valid_gt.copy()
        for surface in surfaces.values():
            valid &= np.isfinite(surface)
        if valid.mean() < 0.5:
            raise ValueError(f"Insufficient paired coverage: {region}")
        for method, surface in surfaces.items():
            result = {"region": region, "method": method}
            result.update(score(z, surface, valid))
            results.append(result)
            profile.update(dtype="float32", nodata=-9999,
                           compress="deflate", predictor=3)
            pred_path = BASE / "predictions" / method / f"{region}_prediction.tif"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(pred_path, "w", **profile) as ds:
                ds.write(np.where(valid, surface, -9999).astype(np.float32), 1)
        print(region, int(valid.sum()), flush=True)
    with (BASE / "region_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)
    print("Saved 60 paired method-region records", flush=True)


if __name__ == "__main__":
    main()
