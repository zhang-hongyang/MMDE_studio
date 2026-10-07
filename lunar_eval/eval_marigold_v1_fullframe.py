"""Full-frame Marigold v1.1 pilot at 32 m and 16 m on five NAC regions.

The experiment intentionally avoids spatial tiling.  Predictions remain relative
depth; oracle affine alignment is reported only as a shape diagnostic.
"""

import csv
import hashlib
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from affine import Affine
from diffusers import MarigoldDepthPipeline
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import torch


BASE = Path(__file__).parent
SOURCE = Path("/mnt/d/nac/official_rdr/expanded")
OUT = Path("/mnt/d/nac/evaluation_20260914/marigold_v1_fullframe_pilot")
MODEL = BASE / "models/marigold_v1_1"
TILED = BASE / "outputs/marigold_v1_1_input256"
REGIONS = ("TRANQPIT1", "ARISTPLAT1", "FRESH1", "RANGER9", "RIMASHARP3")
RESOLUTIONS = ((32, 16), (16, 8))  # output m/px, scale from the 2 m source
NODATA = -9999.0


def read_grid(region, factor):
    folder = SOURCE / region
    dtm_path = folder / f"NAC_DTM_{region}.TIF"
    image_path = next(folder.glob("*_2M.TIF"))
    with rasterio.open(dtm_path) as dtm, rasterio.open(image_path) as image:
        width = (dtm.width + factor - 1) // factor
        height = (dtm.height + factor - 1) // factor
        transform = dtm.transform * Affine.scale(factor, factor)
        kwargs = dict(crs=dtm.crs, transform=transform, width=width, height=height)
        with WarpedVRT(dtm, **kwargs, resampling=Resampling.average) as vrt:
            z = vrt.read(1, masked=True)
        with WarpedVRT(image, **kwargs, resampling=Resampling.average) as vrt:
            gray = vrt.read(1)
        profile = dtm.profile.copy()
        profile.update(width=width, height=height, transform=transform,
                       dtype="float32", nodata=NODATA, compress="deflate",
                       predictor=3)
    return gray.astype(np.uint8), z, profile


def affine_align(raw, target):
    mask = (~np.ma.getmaskarray(target) & np.isfinite(target.data) &
            np.isfinite(raw) & (raw != NODATA))
    x = raw[mask].astype(np.float64)
    y = target.data[mask].astype(np.float64)
    design = np.column_stack((x, np.ones_like(x)))
    scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    aligned = scale * raw.astype(np.float64) + offset
    error = aligned[mask] - y
    return aligned, mask, float(scale), float(offset), float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error**2)))


def write_tif(path, array, profile, valid):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.where(valid, array, NODATA).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def resample_tiled(region, profile):
    with rasterio.open(TILED / f"{region}_prediction.tif") as src:
        with WarpedVRT(src, crs=profile["crs"], transform=profile["transform"],
                       width=profile["width"], height=profile["height"],
                       resampling=Resampling.average) as vrt:
            return vrt.read(1)


def make_figure(region, gray, target, panels, errors, metrics):
    valid = ~np.ma.getmaskarray(target)
    z = target.data
    zmin, zmax = np.percentile(z[valid], (2, 98))
    emax = max(np.percentile(e[valid], 95) for e in errors.values())
    fig, axes = plt.subplots(1, 8, figsize=(25, 4.6), constrained_layout=True)
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("NAC · 32 m/px")
    imz = axes[1].imshow(np.ma.array(z, mask=~valid), cmap="terrain", vmin=zmin, vmax=zmax)
    axes[1].set_title("Official DTM")
    labels = ("Tiled 512→256", "Full frame · 32 m", "Full frame · 16→32 m")
    for ax, label in zip(axes[2:5], labels):
        ax.imshow(np.ma.array(panels[label], mask=~valid), cmap="terrain", vmin=zmin, vmax=zmax)
        ax.set_title(f"{label}\nMAE {metrics[label]:.2f} m")
    ime = None
    for ax, label in zip(axes[5:], labels):
        ime = ax.imshow(np.ma.array(errors[label], mask=~valid), cmap="magma", vmin=0, vmax=emax)
        ax.set_title(f"{label}\nabsolute error")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(imz, ax=axes[1:5], location="bottom", fraction=.06, pad=.03,
                 label="Oracle-affine aligned elevation (m)")
    fig.colorbar(ime, ax=axes[5:], location="bottom", fraction=.06, pad=.03,
                 label="Absolute error (m)")
    fig.suptitle(f"{region} · full-frame Marigold v1.1 diagnostic; no internal tiling", fontsize=13)
    destination = OUT / "comparisons_png" / f"{region}_fullframe_comparison.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = MarigoldDepthPipeline.from_pretrained(
        MODEL, variant="fp16", torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    rows = []
    cache = {}
    for region in REGIONS:
        for resolution, factor in RESOLUTIONS:
            gray, target, profile = read_grid(region, factor)
            rgb = Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2))
            digest = hashlib.sha256(f"{region}-{resolution}".encode()).digest()
            generator = torch.Generator(device="cuda").manual_seed(
                2025 + int.from_bytes(digest[:4], "big")
            )
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = pipe(rgb, num_inference_steps=1, ensemble_size=1,
                          processing_resolution=0, match_input_resolution=True,
                          generator=generator)
            seconds = time.perf_counter() - start
            raw = np.asarray(result.prediction[0, :, :, 0], dtype=np.float32)
            aligned, valid, scale, offset, mae, rmse = affine_align(raw, target)
            write_tif(OUT / "prediction_geotiff" /
                      f"{region}_marigold_v1_full_{resolution}m_raw.tif",
                      raw, profile, valid)
            rows.append(dict(region=region, resolution_m=resolution,
                             width=profile["width"], height=profile["height"],
                             pixels=int(valid.sum()), oracle_affine_scale=scale,
                             oracle_affine_offset=offset, oracle_mae_m=mae,
                             oracle_rmse_m=rmse, inference_seconds=seconds,
                             peak_vram_mib=torch.cuda.max_memory_allocated()/2**20))
            cache[(region, resolution)] = (gray, target, profile, raw)
            print(f"{region} {resolution}m {profile['width']}x{profile['height']} "
                  f"MAE={mae:.3f} time={seconds:.1f}s", flush=True)

    with (OUT / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    # Put all methods on the same 32 m grid for direct paired visualization.
    for region in REGIONS:
        gray, target, profile, raw32 = cache[(region, 32)]
        tiled = resample_tiled(region, profile)
        _, _, profile16, _ = cache[(region, 16)]
        path16 = OUT / "prediction_geotiff" / f"{region}_marigold_v1_full_16m_raw.tif"
        with rasterio.open(path16) as src:
            with WarpedVRT(src, crs=profile["crs"], transform=profile["transform"],
                           width=profile["width"], height=profile["height"],
                           resampling=Resampling.average) as vrt:
                raw16to32 = vrt.read(1)
        names_raw = {"Tiled 512→256": tiled, "Full frame · 32 m": raw32,
                     "Full frame · 16→32 m": raw16to32}
        panels, errors, maes = {}, {}, {}
        for name, raw in names_raw.items():
            aligned, valid, _, _, mae, _ = affine_align(raw, target)
            panels[name] = aligned
            errors[name] = np.abs(aligned - target.data)
            maes[name] = mae
        make_figure(region, gray, target, panels, errors, maes)

    print(f"Outputs: {OUT}", flush=True)


if __name__ == "__main__":
    main()
