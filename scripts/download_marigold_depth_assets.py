#!/usr/bin/env python3
"""Download only the assets required by MMDE's Marigold V2 depth inference."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


ATTEMPTS = 4
REPOSITORIES = (
    (
        "Qwen/Qwen-Image-Edit-2509",
        "Qwen-Image-Edit-2509",
        ["transformer/*", "vae/*"],
    ),
    (
        "huawei-bayerlab/marigold-v2-0",
        "Marigold-V2",
        [
            "depth/Log-stage2/*",
            "qwen_text_embeddings/qwen_edit_2509_qwen_depth_realimg512_*",
        ],
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

    from huggingface_hub import snapshot_download

    checkpoints = args.assets_dir.expanduser().resolve() / "checkpoints"
    for repository, directory, patterns in REPOSITORIES:
        destination = checkpoints / directory
        for attempt in range(1, ATTEMPTS + 1):
            print(
                f"[download] {repository} -> {destination} "
                f"(attempt {attempt}/{ATTEMPTS})",
                flush=True,
            )
            try:
                snapshot_download(
                    repo_id=repository,
                    repo_type="model",
                    local_dir=str(destination),
                    allow_patterns=patterns,
                    max_workers=4,
                )
                break
            except Exception as error:
                if attempt == ATTEMPTS:
                    raise RuntimeError(
                        f"Failed to download {repository} after {ATTEMPTS} attempts"
                    ) from error
                delay = 2**attempt
                print(f"[retry] {error!r}; resuming in {delay}s", flush=True)
                time.sleep(delay)

    required = (
        checkpoints / "Qwen-Image-Edit-2509/transformer/config.json",
        checkpoints / "Qwen-Image-Edit-2509/transformer/diffusion_pytorch_model.safetensors.index.json",
        checkpoints / "Qwen-Image-Edit-2509/vae/config.json",
        checkpoints / "Qwen-Image-Edit-2509/vae/diffusion_pytorch_model.safetensors",
        checkpoints / "Marigold-V2/depth/Log-stage2/trainables.safetensors",
        checkpoints
        / "Marigold-V2/qwen_text_embeddings/qwen_edit_2509_qwen_depth_realimg512_prompt_embeds.pt",
        checkpoints
        / "Marigold-V2/qwen_text_embeddings/qwen_edit_2509_qwen_depth_realimg512_prompt_mask.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required Marigold depth assets are missing:\n" + "\n".join(missing))
    print("Marigold V2 depth assets are complete", flush=True)


if __name__ == "__main__":
    main()
