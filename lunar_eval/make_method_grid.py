"""Make one paired multi-method PNG per region, including only complete methods."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from make_comparisons import DATA, SIDE, select_window


BASE = Path(__file__).parent
OUT = Path("/mnt/d/nac/evaluation_20260914/comparisons_multimethod_png")
METHODS = [
    ("DAV2 · 256px", BASE / "outputs/dav2_small_input256"),
    ("Marigold V1.1 · 256px", BASE / "outputs/marigold_v1_1_input256"),
    ("Marigold V2 · 256px", BASE / "outputs/marigold_v2_input256"),
]


def rows(path):
    with path.open(newline="") as file:
        return {row["region"]: row for row in csv.DictReader(file)}


def complete_methods(regions):
    result = []
    for label, folder in METHODS:
        metrics_path = folder / "region_metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = rows(metrics_path)
        if (set(metrics) != regions or
                any(row["complete"] != "True" for row in metrics.values()) or
                any(not (folder / f"{region}_prediction.tif").exists()
                    for region in regions)):
            continue
        result.append((label, folder, metrics))
    if not result:
        raise ValueError("No all-region method is ready")
    return result


def make_region(region, baseline, methods):
    source = DATA / region
    dtm_path = source / f"NAC_DTM_{region}.TIF"
    image_path = next(source.glob("*_2M.TIF"))
    with rasterio.open(dtm_path) as dtm, rasterio.open(image_path) as source_image:
        window, coverage = select_window(dtm)
        ground = dtm.read(1, window=window, masked=True)
        with WarpedVRT(source_image, crs=dtm.crs, transform=dtm.transform,
                       width=dtm.width, height=dtm.height,
                       resampling=Resampling.bilinear) as image_ds:
            image = image_ds.read(1, window=window)
    z = ground.data.astype(np.float32)
    valid = ~np.ma.getmaskarray(ground) & np.isfinite(z)
    method_panels = []
    for label, folder, metrics in methods:
        with rasterio.open(folder / f"{region}_prediction.tif") as ds:
            raw = ds.read(1, window=window, masked=True)
        valid &= ~np.ma.getmaskarray(raw) & np.isfinite(raw.data)
        aligned = (float(metrics[region]["oracle_affine_scale"]) * raw.data +
                   float(metrics[region]["oracle_affine_offset"]))
        method_panels.append((label, aligned, float(metrics[region]["oracle_mae_m"])))
    if not valid.any():
        raise ValueError(f"No common valid pixels in {region}")
    constant = np.full_like(z, float(np.mean(z[valid], dtype=np.float64)))
    elev = [("DTM", z), ("Constant", constant)]
    elev += [(name, surface) for name, surface, _ in method_panels]
    error = [("Constant error", np.abs(constant - z))]
    error += [(f"{name} error", np.abs(surface - z))
              for name, surface, _ in method_panels]
    zmin, zmax = np.percentile(z[valid], (2, 98))
    emax = max([np.percentile(e[valid], 95) for _, e in error] + [1.0])
    terrain = plt.get_cmap("terrain").copy()
    terrain.set_bad("#171717")
    magma = plt.get_cmap("magma").copy()
    magma.set_bad("#171717")
    panels = [("NAC 2 m/px", image, "gray", None, None)]
    panels += [(name, np.ma.array(surface, mask=~valid), terrain, zmin, zmax)
               for name, surface in elev]
    panels += [(name, np.ma.array(surface, mask=~valid), magma, 0, emax)
               for name, surface in error]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(3.15 * len(panels), 4.6),
                             constrained_layout=True)
    for ax, (name, array, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    first_error = 3 + len(method_panels)
    fig.colorbar(axes[1].images[0], ax=axes[1:first_error],
                 location="bottom", fraction=.045, pad=.04, label="Elevation (m)")
    fig.colorbar(axes[first_error].images[0], ax=axes[first_error:],
                 location="bottom", fraction=.045, pad=.04,
                 label="Absolute error (m)")
    mae = [f"constant {float(baseline[region]['oracle_constant_mae_m']):.2f}"]
    mae += [f"{name} {value:.2f}" for name, _, value in method_panels]
    fig.suptitle(
        f"{region} | same 2.048 km crop, valid {coverage:.0%} | region MAE (m): "
        + "; ".join(mae) + "\nAll model surfaces are oracle-affine aligned "
        "using this test region's DTM. Shape diagnostic, NOT absolute elevation.",
        fontsize=10,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{region}_multimethod.png"
    fig.savefig(dest, dpi=135, facecolor="white")
    plt.close(fig)
    print(dest, flush=True)


def main():
    baseline = rows(BASE / "outputs/dav2_small/constant_metrics.csv")
    regions = set(baseline)
    methods = complete_methods(regions)
    print("Included methods:", [item[0] for item in methods], flush=True)
    for region in sorted(regions):
        make_region(region, baseline, methods)


if __name__ == "__main__":
    main()
