"""Evaluate Marigold V1.1 with 256 px information input on all NAC regions."""

import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import MarigoldDepthPipeline
from PIL import Image

from eval_dav2 import ROOT, evaluate_region


BASE = Path(__file__).parent
MODEL = BASE / "models/marigold_v1_1"
OUT = BASE / "outputs/marigold_v1_1_input256"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = MarigoldDepthPipeline.from_pretrained(
        MODEL, variant="fp16", torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device="cuda").manual_seed(2025)

    def predict(tile):
        rgb = np.repeat(tile[:, :, None], 3, axis=2)
        image = Image.fromarray(rgb).resize((256, 256), Image.Resampling.LANCZOS)
        output = pipe(image, num_inference_steps=1, ensemble_size=1,
                      processing_resolution=256, generator=generator,
                      match_input_resolution=True)
        depth = torch.from_numpy(output.prediction[0, :, :, 0]).float()[None, None]
        depth = F.interpolate(depth, size=tile.shape, mode="bicubic",
                              align_corners=False)
        return depth[0, 0].numpy()

    metrics_path = OUT / ("pilot_metrics.csv" if args.max_tiles is not None
                          else "region_metrics.csv")
    completed = set()
    if metrics_path.exists():
        with metrics_path.open(newline="") as file:
            completed = {row["region"] for row in csv.DictReader(file)
                         if row["complete"] == "True"}
    for region in sorted(ROOT.iterdir()):
        if (not region.is_dir() or region.name in completed or
                (args.region and region.name != args.region)):
            continue
        result = evaluate_region(region, None, None, 512, args.max_tiles, OUT,
                                 predictor=predict)
        with metrics_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(result)
        print(result, flush=True)


if __name__ == "__main__":
    main()
