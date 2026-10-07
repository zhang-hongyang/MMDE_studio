"""Materialize 512 px training tiles (uint8 ortho, float32 DTM, bool mask).

The 2M orthophoto is bilinear-warped onto the DTM grid via WarpedVRT, exactly
as in eval_dav2.py, so training tiles are pixel-aligned with the evaluation
reference grid. Tiles with <10% valid DTM pixels are skipped (same threshold
as the zero-shot evaluation). Per-region georeference is stored alongside the
tiles so predictions can be written back as GeoTIFFs on the identical grid.
"""

import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

ROOT = Path("/mnt/d/nac/official_rdr/expanded")
OUT = Path("/mnt/d/nac/supervised_v1")
TILE = 512
MIN_VALID_FRAC = 0.10


def process_region(region: Path):
    dtm_path = region / f"NAC_DTM_{region.name}.TIF"
    image_path = next(region.glob("*_2M.TIF"))
    region_out = OUT / region.name
    tiles_dir = region_out / "tiles"
    if (region_out / "georef.json").exists():
        print(f"{region.name}: already materialized, skipping", flush=True)
        return
    tiles_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dtm_path) as dtm, rasterio.open(image_path) as src:
        image = WarpedVRT(src, crs=dtm.crs, transform=dtm.transform,
                          width=dtm.width, height=dtm.height,
                          resampling=Resampling.bilinear)
        windows = [Window(x, y, min(TILE, dtm.width - x),
                          min(TILE, dtm.height - y))
                   for y in range(0, dtm.height, TILE)
                   for x in range(0, dtm.width, TILE)]
        index_map = []
        for idx, w in enumerate(windows):
            img = image.read(1, window=w)
            gt = dtm.read(1, window=w, masked=True)
            valid = ~np.ma.getmaskarray(gt) & np.isfinite(gt.data)
            if valid.mean() < MIN_VALID_FRAC:
                continue
            key = f"{int(w.row_off)}_{int(w.col_off)}"
            np.save(tiles_dir / f"{key}_img.npy", img.astype(np.uint8))
            np.save(tiles_dir / f"{key}_dtm.npy", gt.data.astype(np.float32))
            np.save(tiles_dir / f"{key}_mask.npy", valid)
            index_map.append({"key": key, "row": int(w.row_off),
                              "col": int(w.col_off), "valid_frac": float(valid.mean())})
        georef = {
            "crs": dtm.crs.to_string(),
            "transform": list(dtm.transform),
            "height": dtm.height, "width": dtm.width,
            "nodata": -9999,
        }
    (region_out / "georef.json").write_text(json.dumps(georef))
    (region_out / "tiles.json").write_text(json.dumps(index_map))
    print(f"{region.name}: {len(index_map)}/{len(windows)} tiles kept", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for region in sorted(ROOT.iterdir()):
        if not region.is_dir():
            continue
        process_region(region)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
