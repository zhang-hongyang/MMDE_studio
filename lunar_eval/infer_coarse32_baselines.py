"""Run DAV2 or Marigold V1.1 on the exact 32 m / 256 px V2 input tiles."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


BASE = Path(__file__).parent
TRACK = BASE / "outputs/coarse32"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("dav2", "marigold_v1"), required=True)
    args = parser.parse_args()
    with (TRACK / "manifest.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    output_dir = TRACK / f"{args.model}_npy"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "dav2":
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        model_path = BASE / "models/depth_anything_v2_small"
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForDepthEstimation.from_pretrained(model_path).cuda().eval()
    else:
        from diffusers import MarigoldDepthPipeline
        model = MarigoldDepthPipeline.from_pretrained(
            BASE / "models/marigold_v1_1", variant="fp16",
            torch_dtype=torch.float16, local_files_only=True,
        ).to("cuda")
        model.set_progress_bar_config(disable=True)
    for index, row in enumerate(rows, 1):
        destination = output_dir / Path(row["file"]).with_suffix(".npy")
        if destination.exists():
            continue
        image_path = TRACK / "v2_inputs" / f"batch{row['batch']}" / row["file"]
        image = Image.open(image_path).convert("RGB")
        if args.model == "dav2":
            values = processor(images=image, return_tensors="pt")
            values = {key: value.cuda() for key, value in values.items()}
            with torch.inference_mode():
                pred = model(**values).predicted_depth[:, None]
                pred = F.interpolate(pred, size=(256, 256), mode="bicubic",
                                     align_corners=False)
            array = pred[0, 0].float().cpu().numpy()
        else:
            digest = hashlib.sha256(row["file"].encode()).digest()
            seed = 2025 + int.from_bytes(digest[:4], "big")
            generator = torch.Generator(device="cuda").manual_seed(seed)
            result = model(image, num_inference_steps=1, ensemble_size=1,
                           processing_resolution=256, match_input_resolution=True,
                           generator=generator)
            array = result.prediction[0, :, :, 0]
        np.save(destination, np.asarray(array, dtype=np.float32))
        if index % 20 == 0 or index == len(rows):
            print(f"{args.model}: {index}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
