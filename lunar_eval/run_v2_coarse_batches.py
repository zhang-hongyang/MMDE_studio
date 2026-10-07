"""Resumable 4-batch official Marigold V2 inference on the 32 m track.

The defaults preserve the original workstation layout.  A Research Hub run can
keep the immutable inputs elsewhere and writes every generated file under its
per-run ``RESEARCH_HUB_OUTPUT_DIR``.
"""

import argparse
from contextlib import ExitStack
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
INPUT_TRACK = Path(
    os.environ.get("MMDE_COARSE32_INPUT_ROOT", BASE / "outputs/coarse32")
).expanduser().resolve()
OUTPUT_TRACK = Path(
    os.environ.get(
        "RESEARCH_HUB_OUTPUT_DIR",
        os.environ.get("MMDE_OUTPUT_ROOT", str(INPUT_TRACK)),
    )
).expanduser().resolve()
REPO = Path(
    os.environ.get("MMDE_MARIGOLD_REPO", PROJECT_ROOT / "algorithms/marigold-v2")
).expanduser().resolve()
PYTHON = Path(os.environ.get("MMDE_MARIGOLD_PYTHON", sys.executable)).expanduser().resolve()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Deterministically run only the first N tiles; default runs all 83.",
    )
    args = parser.parse_args()
    if args.max_tiles is not None and args.max_tiles < 1:
        parser.error("--max-tiles must be positive")
    return args


def main():
    args = parse_args()
    if not REPO.is_dir():
        raise FileNotFoundError(f"Marigold repository does not exist: {REPO}")
    if not PYTHON.is_file():
        raise FileNotFoundError(f"Marigold Python does not exist: {PYTHON}")
    OUTPUT_TRACK.mkdir(parents=True, exist_ok=True)

    batch_images = []
    for batch in range(4):
        images = INPUT_TRACK / "v2_inputs" / f"batch{batch}"
        selected = sorted(images.glob("*.png"))
        if not selected:
            raise ValueError(f"Empty input batch: {images}")
        batch_images.append(selected)
    available = sum(len(images) for images in batch_images)
    if available != 83:
        raise ValueError(f"Expected 83 input tiles, found {available}")
    target = args.max_tiles or available
    if target > available:
        raise ValueError(f"--max-tiles={target} exceeds input population {available}")

    total_expected = total_actual = 0
    selected_names = []
    for batch in range(4):
        images = INPUT_TRACK / "v2_inputs" / f"batch{batch}"
        remaining = target - total_expected
        selected = batch_images[batch][:remaining]
        if not selected:
            break
        output = OUTPUT_TRACK / f"v2_batch{batch}"
        predictions = output / "images/predictions_npy"
        expected = len(selected)
        actual = sum((predictions / f"{image.stem}.npy").is_file() for image in selected)
        if actual != expected:
            with ExitStack() as stack:
                inference_images = images
                if expected != len(batch_images[batch]):
                    temporary = Path(
                        stack.enter_context(
                            tempfile.TemporaryDirectory(prefix=f"mmde-batch{batch}-")
                        )
                    )
                    for image in selected:
                        (temporary / image.name).symlink_to(image)
                    inference_images = temporary
                command = [str(PYTHON), str(REPO / "scripts/infer.py"),
                           "--image_dir", str(inference_images),
                           "--output_dir", str(output),
                           "--width", "256", "--height", "256"]
                log_path = OUTPUT_TRACK / f"v2_batch{batch}.log"
                print(f"batch{batch}: {actual}/{expected}; starting", flush=True)
                with log_path.open("w") as log:
                    result = subprocess.run(
                        command,
                        cwd=REPO,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
            if result.returncode != 0:
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(
                    f"Marigold batch{batch} exited with {result.returncode}:\n{log_tail}"
                )
            actual = sum((predictions / f"{image.stem}.npy").is_file() for image in selected)
            if actual != expected:
                raise RuntimeError(f"Incomplete batch{batch}: {actual}/{expected}")
        print(f"batch{batch}: complete {actual}/{expected}", flush=True)
        total_expected += expected
        total_actual += actual
        selected_names.extend(str(image.relative_to(INPUT_TRACK)) for image in selected)
    if total_expected != target or total_actual != target:
        raise ValueError(
            f"Expected {target} complete tiles, found {total_actual}/{total_expected}"
        )
    metrics = [
        {"name": "expected_tiles", "value": total_expected},
        {"name": "completed_tiles", "value": total_actual},
        {"name": "completion_ratio", "value": total_actual / total_expected},
    ]
    (OUTPUT_TRACK / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_TRACK / "hub-run-config.json").write_text(
        json.dumps(
            {
                "inputRoot": str(INPUT_TRACK),
                "outputRoot": str(OUTPUT_TRACK),
                "maxTiles": args.max_tiles,
                "selectedTiles": selected_names,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"All {target} V2 coarse tiles complete", flush=True)


if __name__ == "__main__":
    main()
