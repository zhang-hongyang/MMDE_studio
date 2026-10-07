"""Region-grouped, 2 m-grid DAV2 evaluation on the expanded NAC batch."""

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


ROOT = Path("/mnt/d/nac/official_rdr/expanded")
MODEL = Path(__file__).parent / "models/depth_anything_v2_small"


def predict(tile, processor, model, input_size=None):
    rgb = np.repeat(tile[:, :, None], 3, axis=2)
    image = Image.fromarray(rgb)
    if input_size is not None:
        image = image.resize((input_size, input_size), Image.Resampling.LANCZOS)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.cuda() for key, value in inputs.items()}
    with torch.inference_mode():
        depth = model(**inputs).predicted_depth[:, None]
        depth = F.interpolate(depth, size=tile.shape, mode="bicubic", align_corners=False)
    return depth[0, 0].float().cpu().numpy()


def evaluate_region(region, processor, model, tile_size, max_tiles, out_dir,
                    predictor=None):
    dtm_path = region / f"NAC_DTM_{region.name}.TIF"
    image_path = next(region.glob("*_2M.TIF"))
    prediction_path = out_dir / f"{region.name}_prediction.tif"
    with rasterio.open(dtm_path) as dtm, rasterio.open(image_path) as source_image:
        image = WarpedVRT(source_image, crs=dtm.crs, transform=dtm.transform,
                          width=dtm.width, height=dtm.height,
                          resampling=Resampling.bilinear)
        profile = dtm.profile.copy()
        profile.update(dtype="float32", nodata=-9999, compress="deflate", predictor=3)
        # Fixed 512 grid, independent of source TIFF storage block layout.
        windows = [rasterio.windows.Window(x, y, min(tile_size, dtm.width - x),
                                         min(tile_size, dtm.height - y))
                   for y in range(0, dtm.height, tile_size)
                   for x in range(0, dtm.width, tile_size)]
        if max_tiles is not None:
            windows = windows[:max_tiles]
        with rasterio.open(prediction_path, "w", **profile) as prediction:
            for index, window in enumerate(windows, 1):
                img = image.read(1, window=window)
                gt = dtm.read(1, window=window, masked=True)
                valid = ~np.ma.getmaskarray(gt) & np.isfinite(gt.data)
                if valid.mean() < 0.1:
                    prediction.write(np.full(img.shape, -9999, dtype=np.float32),
                                     1, window=window)
                    continue
                pred = (predictor(img) if predictor is not None
                        else predict(img, processor, model))
                pred[~valid] = -9999
                prediction.write(pred, 1, window=window)
                if index % 50 == 0:
                    print(f"{region.name}: {index}/{len(windows)} tiles", flush=True)
        # Compute one oracle affine fit per region. This is a shape diagnostic,
        # not deployable absolute lunar elevation accuracy.
        n = sx = sy = sxx = sxy = 0.0
        with rasterio.open(prediction_path) as prediction:
            for window in windows:
                pred = prediction.read(1, window=window)
                gt = dtm.read(1, window=window, masked=True)
                valid = (pred != -9999) & ~np.ma.getmaskarray(gt) & np.isfinite(gt.data)
                if not valid.any():
                    continue
                x = pred[valid].astype(np.float64)
                y = gt.data[valid].astype(np.float64)
                n += len(x)
                sx += x.sum(); sy += y.sum(); sxx += x @ x; sxy += x @ y
            denominator = sxx - sx * sx / n
            scale = (sxy - sx * sy / n) / denominator if denominator > 0 else 0.0
            offset = (sy - scale * sx) / n
            ae = se = 0.0
            for window in windows:
                pred = prediction.read(1, window=window)
                gt = dtm.read(1, window=window, masked=True)
                valid = (pred != -9999) & ~np.ma.getmaskarray(gt) & np.isfinite(gt.data)
                if not valid.any():
                    continue
                error = scale * pred[valid].astype(np.float64) + offset - gt.data[valid]
                ae += np.abs(error).sum(); se += error @ error
    return {"region": region.name, "tiles_processed": len(windows),
            "pixels": int(n), "oracle_affine_scale": scale,
            "oracle_affine_offset": offset, "oracle_mae_m": ae / n,
            "oracle_rmse_m": np.sqrt(se / n), "complete": max_tiles is None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Diagnostic only; omit for full-region evaluation")
    parser.add_argument("--region", default=None)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--input-size", type=int, default=None,
                        help="Pre-resize source tile before the model processor")
    args = parser.parse_args()
    suffix = "dav2_small" if args.input_size is None else f"dav2_small_input{args.input_size}"
    out_dir = Path(__file__).parent / f"outputs/{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).cuda().eval()
    regions = sorted(ROOT.iterdir())
    if args.region:
        regions = [region for region in regions if region.name == args.region]
    results_path = out_dir / "region_metrics.csv"
    completed = set()
    if results_path.exists():
        with results_path.open(newline="") as file:
            completed = {row["region"] for row in csv.DictReader(file)
                         if row["complete"] == "True"}
    for region in regions:
        if not region.is_dir() or region.name in completed:
            continue
        predictor = lambda img: predict(img, processor, model, args.input_size)
        result = evaluate_region(region, processor, model, args.tile_size,
                                 args.max_tiles, out_dir, predictor=predictor)
        with results_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(result)
        print(result, flush=True)


if __name__ == "__main__":
    main()
