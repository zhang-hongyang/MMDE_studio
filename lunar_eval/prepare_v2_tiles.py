"""Prepare 256 px PNGs from paired 512 px / 2 m NAC windows for V2 inference."""

import csv
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


DATA = Path("/mnt/d/nac/official_rdr/expanded")
OUT = Path(__file__).parent / "outputs/marigold_v2_input256/inputs"
MANIFEST = OUT.parent / "tiles_manifest.csv"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for region in sorted(DATA.iterdir()):
        if not region.is_dir():
            continue
        dtm_path = region / f"NAC_DTM_{region.name}.TIF"
        image_path = next(region.glob("*_2M.TIF"))
        with rasterio.open(dtm_path) as dtm, rasterio.open(image_path) as source:
            with WarpedVRT(source, crs=dtm.crs, transform=dtm.transform,
                           width=dtm.width, height=dtm.height,
                           resampling=Resampling.bilinear) as image:
                for y in range(0, dtm.height, 512):
                    for x in range(0, dtm.width, 512):
                        w, h = min(512, dtm.width - x), min(512, dtm.height - y)
                        window = Window(x, y, w, h)
                        gt = dtm.read(1, window=window, masked=True)
                        valid = ~np.ma.getmaskarray(gt) & np.isfinite(gt.data)
                        if valid.mean() < 0.1:
                            continue
                        name = f"{region.name}_y{y:05d}_x{x:05d}_w{w}_h{h}.png"
                        destination = OUT / name
                        if not destination.exists():
                            pixels = image.read(1, window=window)
                            rgb = np.repeat(pixels[:, :, None], 3, axis=2)
                            Image.fromarray(rgb).resize(
                                (256, 256), Image.Resampling.LANCZOS
                            ).save(destination)
                        rows.append({"region": region.name, "file": name,
                                     "x": x, "y": y, "width": w, "height": h,
                                     "valid_fraction": float(valid.mean())})
        print(region.name, sum(row["region"] == region.name for row in rows),
              flush=True)
    with MANIFEST.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"Prepared {len(rows)} input PNGs: {OUT}", flush=True)


if __name__ == "__main__":
    main()
