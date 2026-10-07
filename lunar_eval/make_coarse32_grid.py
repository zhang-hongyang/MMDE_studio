"""Create 20 paired 32 m multi-method PNGs after full-track scoring."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window


BASE = Path(__file__).parent / "outputs/coarse32"
OUT = Path("/mnt/d/nac/evaluation_20260914/comparisons_coarse32_png")
SIDE = 128  # 4.096 km at 32 m/pixel
LABELS = {"dav2": "DAV2", "marigold_v1": "Marigold V1.1",
          "marigold_v2": "Marigold V2"}


def pick_window(ds):
    best = None
    best_fraction = -1
    for yf, xf in ((.5, .5), (.25, .5), (.75, .5), (.5, .25),
                   (.5, .75), (.25, .25), (.25, .75), (.75, .25), (.75, .75)):
        x = min(max(round(ds.width * xf - SIDE / 2), 0), ds.width - SIDE)
        y = min(max(round(ds.height * yf - SIDE / 2), 0), ds.height - SIDE)
        window = Window(x, y, SIDE, SIDE)
        sample = ds.read(1, window=window, masked=True)
        fraction = np.mean(~np.ma.getmaskarray(sample))
        if fraction > best_fraction:
            best, best_fraction = window, fraction
    return best, best_fraction


def make(region, records):
    gt_path = BASE / "reference" / f"{region}_DTM_32m.tif"
    image_path = BASE / "images" / f"{region}_NAC_32m.tif"
    with rasterio.open(gt_path) as ds:
        window, fraction = pick_window(ds)
        z = ds.read(1, window=window, masked=True)
        full = ds.read(1, masked=True)
        mean = float(full.mean())
    with rasterio.open(image_path) as ds:
        image = ds.read(1, window=window)
    valid = ~np.ma.getmaskarray(z) & np.isfinite(z.data)
    surfaces = []
    for method in ("dav2", "marigold_v1", "marigold_v2"):
        path = BASE / "predictions" / method / f"{region}_prediction.tif"
        with rasterio.open(path) as ds:
            raw = ds.read(1, window=window, masked=True)
        valid &= ~np.ma.getmaskarray(raw) & np.isfinite(raw.data)
        record = records[(region, method)]
        surface = (float(record["oracle_affine_scale"]) * raw.data +
                   float(record["oracle_affine_offset"]))
        surfaces.append((LABELS[method], surface))
    constant = np.full_like(z.data, mean)
    zmin, zmax = np.percentile(z.data[valid], (2, 98))
    elevation_panels = [("Official DTM", z.data), ("Constant", constant)] + surfaces
    error_panels = [(f"{label} error", np.abs(surface - z.data))
                    for label, surface in [("Constant", constant)] + surfaces]
    emax = max([np.percentile(e[valid], 95) for _, e in error_panels] + [1.0])
    terrain = plt.get_cmap("terrain").copy(); terrain.set_bad("#171717")
    magma = plt.get_cmap("magma").copy(); magma.set_bad("#171717")
    panels = [("NAC 32 m/px", image, "gray", None, None)]
    panels += [(label, np.ma.array(surface, mask=~valid), terrain, zmin, zmax)
               for label, surface in elevation_panels]
    panels += [(label, np.ma.array(error, mask=~valid), magma, 0, emax)
               for label, error in error_panels]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(2.8 * len(panels), 4.0),
                             constrained_layout=True)
    for ax, (label, array, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    split = 1 + len(elevation_panels)
    fig.colorbar(axes[1].images[0], ax=axes[1:split], location="bottom",
                 fraction=.05, pad=.04, label="Elevation (m)")
    fig.colorbar(axes[split].images[0], ax=axes[split:], location="bottom",
                 fraction=.05, pad=.04, label="Absolute error (m)")
    maes = [f"constant {float(records[(region, 'dav2')]['oracle_constant_mae_m']):.2f}"]
    maes += [f"{LABELS[m]} {float(records[(region, m)]['oracle_mae_m']):.2f}"
             for m in ("dav2", "marigold_v1", "marigold_v2")]
    fig.suptitle(
        f"{region} | same 4.096 km crop at 32 m/px, valid {fraction:.0%} | "
        "full-region oracle MAE (m): " + "; ".join(maes) + "\n"
        "All predictions share the 32 m grid and are test-region affine-aligned. "
        "NOT 2 m or deployable absolute elevation.", fontsize=10,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{region}_coarse32_multimethod.png"
    fig.savefig(destination, dpi=135, facecolor="white")
    plt.close(fig)
    print(destination, flush=True)


def main():
    with (BASE / "region_metrics.csv").open(newline="") as file:
        records = {(row["region"], row["method"]): row
                   for row in csv.DictReader(file)}
    regions = sorted({region for region, _ in records})
    if len(regions) != 20 or len(records) != 60:
        raise ValueError("Need complete 20-region / 3-method scores")
    for region in regions:
        make(region, records)


if __name__ == "__main__":
    main()
