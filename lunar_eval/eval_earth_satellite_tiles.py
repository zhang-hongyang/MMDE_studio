"""Apply the lunar 512->256 non-overlapping tile protocol to one Earth image."""

import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
import torch
import torch.nn.functional as F


BASE = Path(__file__).parent
SOURCE = Path("/mnt/e/RSdata/quake_data/res/E036.9_N37.6/block_image/row1000_col1000/left_image_1_1.tif")
OUT = SOURCE.parent / "depth_model_comparison_512tiles"
TILE = 512
NET = 256
NAMES = [f"tile_y{y:04d}_x{x:04d}" for y in range(0, 1536, TILE) for x in range(0, 1536, TILE)]


def load_and_pad():
    with rasterio.open(SOURCE) as ds:
        image = ds.read(1)
    h, w = image.shape
    padded = np.pad(image, ((0, 1536-h), (0, 1536-w)), mode="reflect")
    return image, padded


def prepare_inputs(padded):
    folder = OUT / "inputs_256"
    folder.mkdir(parents=True, exist_ok=True)
    index = 0
    for y in range(0, 1536, TILE):
        for x in range(0, 1536, TILE):
            patch = padded[y:y+TILE, x:x+TILE]
            rgb = np.repeat(patch[:, :, None], 3, axis=2)
            Image.fromarray(rgb).resize((NET, NET), Image.Resampling.LANCZOS).save(folder / f"{NAMES[index]}.png")
            index += 1


def run_dav2():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(BASE / "models/depth_anything_v2_small")
    model = AutoModelForDepthEstimation.from_pretrained(BASE / "models/depth_anything_v2_small").cuda().eval()
    destination = OUT / "dav2_npy"; destination.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        image = Image.open(OUT / "inputs_256" / f"{name}.png").convert("RGB")
        values = {k: v.cuda() for k, v in processor(images=image, return_tensors="pt").items()}
        with torch.inference_mode():
            pred = model(**values).predicted_depth[:, None]
            pred = F.interpolate(pred, size=(TILE, TILE), mode="bicubic", align_corners=False)
        np.save(destination / f"{name}.npy", pred[0, 0].float().cpu().numpy().astype(np.float32))
    del model; torch.cuda.empty_cache()


def run_v1():
    from diffusers import MarigoldDepthPipeline
    pipe = MarigoldDepthPipeline.from_pretrained(
        BASE / "models/marigold_v1_1", variant="fp16", torch_dtype=torch.float16,
        local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    destination = OUT / "marigold_v1_npy"; destination.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(NAMES):
        image = Image.open(OUT / "inputs_256" / f"{name}.png").convert("RGB")
        generator = torch.Generator(device="cuda").manual_seed(2025+i)
        result = pipe(image, num_inference_steps=1, ensemble_size=1,
                      processing_resolution=256, match_input_resolution=True,
                      generator=generator)
        pred = torch.from_numpy(result.prediction[0, :, :, 0]).float()[None, None]
        pred = F.interpolate(pred, size=(TILE, TILE), mode="bicubic", align_corners=False)
        np.save(destination / f"{name}.npy", pred[0, 0].numpy().astype(np.float32))
    del pipe; torch.cuda.empty_cache()


def run_v2():
    destination = OUT / "marigold_v2_run"
    expected = destination / "images" / "predictions_npy"
    if len(list(expected.glob("*.npy"))) == len(NAMES):
        return
    command = [str(Path("/home/research/code/MMDE_studio/algorithms/.venv_marigold/bin/python")),
               "/home/research/code/MMDE_studio/algorithms/marigold-v2/scripts/infer.py",
               "--image_dir", str(OUT / "inputs_256"),
               "--output_dir", str(destination), "--width", "256", "--height", "256"]
    with (OUT / "marigold_v2.log").open("w") as log:
        subprocess.run(command, cwd="/home/research/code/MMDE_studio/algorithms/marigold-v2", stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def assemble(folder, v2=False):
    canvas = np.empty((1536, 1536), np.float32)
    for i, name in enumerate(NAMES):
        path = folder / f"{name}.npy"
        tile = np.load(path)
        if tile.ndim > 2:
            tile = np.squeeze(tile)
        if tile.shape != (TILE, TILE):
            tile = np.asarray(Image.fromarray(tile).resize((TILE, TILE), Image.Resampling.BICUBIC))
        y, x = (i // 3) * TILE, (i % 3) * TILE
        canvas[y:y+TILE, x:x+TILE] = tile
    return canvas[:1302, :1300]


def seam_stat(array):
    dx, dy = np.abs(np.diff(array, axis=1)), np.abs(np.diff(array, axis=0))
    boundary = np.r_[dx[:, 511].ravel(), dx[:, 1023].ravel(),
                     dy[511, :].ravel(), dy[1023, :].ravel()]
    nearby = np.r_[dx[:, 495].ravel(), dx[:, 527].ravel(), dx[:, 1007].ravel(), dx[:, 1039].ravel(),
                   dy[495, :].ravel(), dy[527, :].ravel(), dy[1007, :].ravel(), dy[1039, :].ravel()]
    return float(np.median(boundary)), float(np.median(nearby)), float(np.median(boundary)/np.median(nearby))


def save_results(source, predictions):
    rows=[]
    for name, array in predictions.items():
        b, n, ratio = seam_stat(array)
        rows.append({"method": name, "boundary_median_jump": b,
                     "nearby_median_jump": n, "boundary_ratio": ratio})
        np.save(OUT / f"{name}_stitched_raw.npy", array)
    with (OUT / "seam_metrics.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
    axes[0].imshow(source, cmap="gray", vmin=np.percentile(source,1), vmax=np.percentile(source,99))
    axes[0].set_title("Earth satellite input")
    for ax, (name, array), row in zip(axes[1:], predictions.items(), rows):
        lo, hi = np.percentile(array, (2, 98))
        ax.imshow(array, cmap="Spectral", vmin=lo, vmax=hi)
        ax.set_title(f"{name}\n512→256 independent tiles\nseam ratio {row['boundary_ratio']:.1f}×")
    for ax in axes:
        ax.axvline(511.5, color="white", lw=.5, alpha=.5); ax.axvline(1023.5, color="white", lw=.5, alpha=.5)
        ax.axhline(511.5, color="white", lw=.5, alpha=.5); ax.axhline(1023.5, color="white", lw=.5, alpha=.5)
        ax.axis("off")
    fig.savefig(OUT / "earth_satellite_tiled_methods.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source, padded = load_and_pad(); prepare_inputs(padded)
    run_dav2(); run_v1(); run_v2()
    predictions = {
        "DAV2": assemble(OUT / "dav2_npy"),
        "Marigold_V1_1": assemble(OUT / "marigold_v1_npy"),
        "Marigold_V2": assemble(OUT / "marigold_v2_run/images/predictions_npy"),
    }
    save_results(source, predictions)
    print(OUT)


if __name__ == "__main__":
    main()
