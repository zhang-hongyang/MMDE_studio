"""Create a separate 32 m/pixel full-coverage track for all 20 NAC regions."""

import csv
import math
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


SOURCE = Path("/mnt/d/nac/official_rdr/expanded")
BASE = Path(__file__).parent / "outputs/coarse32"
DTM_DIR = BASE / "reference"
IMAGE_DIR = BASE / "images"
INPUT_DIR = BASE / "v2_inputs"
FACTOR = 16  # 2 m source -> 32 m track; do not claim 2 m accuracy
TILE = 256


def write_tif(destination, array, profile):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as ds:
        ds.write(array, 1)


def main():
    DTM_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, region in enumerate(sorted(SOURCE.iterdir())):
        if not region.is_dir():
            continue
        dtm_path = region / f"NAC_DTM_{region.name}.TIF"
        image_path = next(region.glob("*_2M.TIF"))
        with rasterio.open(dtm_path) as dtm, rasterio.open(image_path) as source_image:
            width = math.ceil(dtm.width / FACTOR)
            height = math.ceil(dtm.height / FACTOR)
            transform = dtm.transform * Affine.scale(FACTOR, FACTOR)
            with WarpedVRT(dtm, crs=dtm.crs, transform=transform,
                           width=width, height=height,
                           resampling=Resampling.average) as coarse_dtm:
                z = coarse_dtm.read(1, masked=True)
            with WarpedVRT(source_image, crs=dtm.crs, transform=transform,
                           width=width, height=height,
                           resampling=Resampling.average) as coarse_image:
                gray = coarse_image.read(1)
            reference = np.where(np.ma.getmaskarray(z), -9999,
                                 z.data).astype(np.float32)
            profile = dtm.profile.copy()
            profile.update(width=width, height=height, transform=transform,
                           dtype="float32", nodata=-9999,
                           compress="deflate", predictor=3)
            write_tif(DTM_DIR / f"{region.name}_DTM_32m.tif", reference, profile)
            image_profile = profile.copy()
            image_profile.update(dtype="uint8", nodata=None, predictor=2)
            write_tif(IMAGE_DIR / f"{region.name}_NAC_32m.tif",
                      gray.astype(np.uint8), image_profile)
            for y in range(0, height, TILE):
                for x in range(0, width, TILE):
                    w = min(TILE, width - x)
                    h = min(TILE, height - y)
                    patch = gray[y:y+h, x:x+w]
                    # Reflect-pad to a 256 square without changing ground scale.
                    padded = np.pad(patch, ((0, TILE-h), (0, TILE-w)),
                                    mode="reflect")
                    rgb = np.repeat(padded[:, :, None], 3, axis=2)
                    name = f"{region.name}_y{y:04d}_x{x:04d}.png"
                    batch = index % 4
                    destination = INPUT_DIR / f"batch{batch}" / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(destination)
                    valid = reference[y:y+h, x:x+w] != -9999
                    rows.append({"region": region.name, "batch": batch,
                                 "file": name, "x": x, "y": y,
                                 "width": w, "height": h,
                                 "valid_fraction": float(valid.mean())})
        print(region.name, width, height, flush=True)
    with (BASE / "manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"Prepared {len(rows)} coarse tiles (32 m/pixel)", flush=True)


if __name__ == "__main__":
    main()
