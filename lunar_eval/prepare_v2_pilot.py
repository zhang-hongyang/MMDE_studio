"""Prepare one reproducible NAC input crop for Marigold V2 memory testing."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


region = Path("/mnt/d/nac/official_rdr/expanded/TRANQPIT1")
dtm_path = region / "NAC_DTM_TRANQPIT1.TIF"
ortho_path = next(region.glob("*_2M.TIF"))
destination = Path(__file__).parent / "pilot_inputs/TRANQPIT1_center_512.png"
destination.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(dtm_path) as dtm, rasterio.open(ortho_path) as source:
    window = Window((dtm.width - 512) // 2, (dtm.height - 512) // 2, 512, 512)
    with WarpedVRT(source, crs=dtm.crs, transform=dtm.transform,
                   width=dtm.width, height=dtm.height,
                   resampling=Resampling.bilinear) as ortho:
        pixels = ortho.read(1, window=window)
    gt = dtm.read(1, window=window, masked=True)
    if np.mean(~np.ma.getmaskarray(gt)) < 0.95:
        raise ValueError("Pilot crop has insufficient DTM coverage")
Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2)).save(destination)
print(destination)
