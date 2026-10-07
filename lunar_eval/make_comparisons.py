"""Make paired, honest multi-column PNGs for every completed NAC region."""

import csv
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


DATA = Path("/mnt/d/nac/official_rdr/expanded")
EVAL = Path(__file__).parent / "outputs/dav2_small"
OUT = Path("/mnt/d/nac/evaluation_20260914/comparisons_png")
SIDE = 1024  # 2 m/pixel, approximately 2.048 km square
MODEL_LABEL = "DAV2"


def read_rows(name):
    with (EVAL / name).open(newline="") as file:
        return {row["region"]: row for row in csv.DictReader(file)}


def candidate_windows(dataset):
    # Center first; only move if the official DTM has materially better coverage.
    for yf, xf in ((.5, .5), (.25, .5), (.75, .5), (.5, .25),
                   (.5, .75), (.25, .25), (.25, .75), (.75, .25), (.75, .75)):
        x = min(max(round(dataset.width * xf - SIDE / 2), 0), dataset.width - SIDE)
        y = min(max(round(dataset.height * yf - SIDE / 2), 0), dataset.height - SIDE)
        yield Window(x, y, SIDE, SIDE)


def select_window(dataset):
    best_window = None
    best_coverage = -1.0
    for window in candidate_windows(dataset):
        sample = dataset.read(1, window=window, out_shape=(128, 128),
                              resampling=Resampling.nearest, masked=True)
        coverage = np.mean(~np.ma.getmaskarray(sample) & np.isfinite(sample.data))
        if coverage > best_coverage:
            best_window, best_coverage = window, coverage
    return best_window, best_coverage


def make_region(region, metric, baseline):
    src = DATA / region
    dtm_path = src / f"NAC_DTM_{region}.TIF"
    image_path = next(src.glob("*_2M.TIF"))
    pred_path = EVAL / f"{region}_prediction.tif"
    with rasterio.open(dtm_path) as dtm, rasterio.open(image_path) as ortho_source, \
         rasterio.open(pred_path) as prediction:
        window, coverage = select_window(dtm)
        with WarpedVRT(ortho_source, crs=dtm.crs, transform=dtm.transform,
                       width=dtm.width, height=dtm.height,
                       resampling=Resampling.bilinear) as ortho:
            image = ortho.read(1, window=window)
        ground = dtm.read(1, window=window, masked=True)
        raw = prediction.read(1, window=window, masked=True)
    valid = (~np.ma.getmaskarray(ground) & ~np.ma.getmaskarray(raw) &
             np.isfinite(ground.data) & np.isfinite(raw.data))
    if not valid.any():
        raise ValueError(f"No paired valid pixels in {region}")
    z = ground.data.astype(np.float32)
    estimate = (float(metric["oracle_affine_scale"]) * raw.data +
                float(metric["oracle_affine_offset"]))
    mean_z = float(np.mean(z[valid], dtype=np.float64))
    constant = np.full_like(z, mean_z)
    err_model = np.abs(estimate - z)
    err_constant = np.abs(constant - z)
    zmin, zmax = np.percentile(z[valid], (2, 98))
    emax = max(np.percentile(err_model[valid], 95),
               np.percentile(err_constant[valid], 95), 1.0)

    relief = plt.get_cmap("terrain").copy()
    relief.set_bad("#171717")
    errors = plt.get_cmap("magma").copy()
    errors.set_bad("#171717")
    fig, axes = plt.subplots(1, 6, figsize=(22, 4.7), constrained_layout=True)
    panels = [
        ("NAC orthophoto (2 m/px)", image, "gray", None, None),
        ("Official stereo DTM", z, relief, zmin, zmax),
        (f"{MODEL_LABEL} · oracle affine", estimate, relief, zmin, zmax),
        ("Oracle constant", constant, relief, zmin, zmax),
        (f"|{MODEL_LABEL} − DTM|", err_model, errors, 0, emax),
        ("|Constant − DTM|", err_constant, errors, 0, emax),
    ]
    for ax, (title, array, cmap, vmin, vmax) in zip(axes, panels):
        shown = np.ma.array(array, mask=~valid) if title != "NAC orthophoto (2 m/px)" else array
        ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(axes[2].images[0], ax=axes[1:4], location="bottom",
                 fraction=0.045, pad=0.04, label="Elevation (m)")
    fig.colorbar(axes[5].images[0], ax=axes[4:6], location="bottom",
                 fraction=0.07, pad=0.04, label="Absolute error (m)")
    fig.suptitle(
        f"{region} | 2.048 km crop | valid {coverage:.0%} | "
        f"region MAE: {MODEL_LABEL} {float(metric['oracle_mae_m']):.2f} m; "
        f"constant {float(baseline['oracle_constant_mae_m']):.2f} m\n"
        f"{MODEL_LABEL} was affine-fitted to this test region's DTM: shape-only oracle, "
        "NOT deployable metric elevation. Same scale in elevation/error pairs.",
        fontsize=11,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{region}_six_column.png"
    fig.savefig(destination, dpi=135, facecolor="white")
    plt.close(fig)
    print(destination, flush=True)


def main():
    global EVAL, OUT, MODEL_LABEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--input256", action="store_true")
    args = parser.parse_args()
    if args.input256:
        EVAL = Path(__file__).parent / "outputs/dav2_small_input256"
        OUT = Path("/mnt/d/nac/evaluation_20260914/comparisons_png_input256")
        MODEL_LABEL = "DAV2 256px"
    metrics = read_rows("region_metrics.csv")
    if args.input256:
        baseline_path = Path(__file__).parent / "outputs/dav2_small/constant_metrics.csv"
        with baseline_path.open(newline="") as file:
            baselines = {row["region"]: row for row in csv.DictReader(file)}
    else:
        baselines = read_rows("constant_metrics.csv")
    if set(metrics) != set(baselines):
        raise ValueError("Paired result sets differ; refusing incomplete figures")
    for region in sorted(metrics):
        if metrics[region]["complete"] != "True":
            continue
        make_region(region, metrics[region], baselines[region])


if __name__ == "__main__":
    main()
