"""Regression test for the Research Hub path contract."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_v2_coarse_batches.py")


class PortableCoarseBatchTest(unittest.TestCase):
    def test_inputs_remain_read_only_and_outputs_use_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_root = root / "input"
            output_root = root / "run-output"
            repository = root / "marigold-v2"
            infer = repository / "scripts/infer.py"
            infer.parent.mkdir(parents=True)
            infer.write_text(
                """\
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=Path, required=True)
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--width')
parser.add_argument('--height')
args = parser.parse_args()
destination = args.output_dir / 'images/predictions_npy'
destination.mkdir(parents=True, exist_ok=True)
for image in args.image_dir.glob('*.png'):
    (destination / f'{image.stem}.npy').write_bytes(b'prediction')
""",
                encoding="utf-8",
            )

            counts = (21, 21, 21, 20)
            for batch, count in enumerate(counts):
                batch_directory = input_root / "v2_inputs" / f"batch{batch}"
                batch_directory.mkdir(parents=True)
                for index in range(count):
                    (batch_directory / f"tile-{index:03d}.png").write_bytes(b"input")

            environment = {
                **os.environ,
                "MMDE_COARSE32_INPUT_ROOT": str(input_root),
                "MMDE_MARIGOLD_REPO": str(repository),
                "MMDE_MARIGOLD_PYTHON": sys.executable,
                "RESEARCH_HUB_OUTPUT_DIR": str(output_root),
            }
            result = subprocess.run(
                [sys.executable, str(SCRIPT)],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertIn("All 83 V2 coarse tiles complete", result.stdout)
            self.assertFalse((input_root / "v2_batch0").exists())
            self.assertEqual(
                len(list(output_root.glob("v2_batch*/images/predictions_npy/*.npy"))),
                83,
            )
            metrics = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics[1], {"name": "completed_tiles", "value": 83})
            self.assertEqual(metrics[2], {"name": "completion_ratio", "value": 1.0})

    def test_subprocess_log_tail_is_visible_in_the_parent_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_root = root / "input"
            repository = root / "marigold-v2"
            infer = repository / "scripts/infer.py"
            infer.parent.mkdir(parents=True)
            infer.write_text("raise RuntimeError('inner model failure')\n", encoding="utf-8")
            for batch, count in enumerate((21, 21, 21, 20)):
                batch_directory = input_root / "v2_inputs" / f"batch{batch}"
                batch_directory.mkdir(parents=True)
                for index in range(count):
                    (batch_directory / f"tile-{index:03d}.png").write_bytes(b"input")
            environment = {
                **os.environ,
                "MMDE_COARSE32_INPUT_ROOT": str(input_root),
                "MMDE_MARIGOLD_REPO": str(repository),
                "MMDE_MARIGOLD_PYTHON": sys.executable,
                "RESEARCH_HUB_OUTPUT_DIR": str(root / "run-output"),
            }

            result = subprocess.run(
                [sys.executable, str(SCRIPT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("inner model failure", result.stderr)

    def test_max_tiles_runs_a_deterministic_smoke_subset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_root = root / "input"
            output_root = root / "run-output"
            repository = root / "marigold-v2"
            infer = repository / "scripts/infer.py"
            infer.parent.mkdir(parents=True)
            infer.write_text(
                """\
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=Path, required=True)
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--width')
parser.add_argument('--height')
args = parser.parse_args()
destination = args.output_dir / 'images/predictions_npy'
destination.mkdir(parents=True, exist_ok=True)
for image in args.image_dir.glob('*.png'):
    (destination / f'{image.stem}.npy').write_bytes(b'prediction')
""",
                encoding="utf-8",
            )
            counts = (21, 21, 21, 20)
            for batch, count in enumerate(counts):
                batch_directory = input_root / "v2_inputs" / f"batch{batch}"
                batch_directory.mkdir(parents=True)
                for index in range(count):
                    (batch_directory / f"tile-{index:03d}.png").write_bytes(b"input")
            environment = {
                **os.environ,
                "MMDE_COARSE32_INPUT_ROOT": str(input_root),
                "MMDE_MARIGOLD_REPO": str(repository),
                "MMDE_MARIGOLD_PYTHON": sys.executable,
                "RESEARCH_HUB_OUTPUT_DIR": str(output_root),
            }

            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--max-tiles", "1"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertIn("All 1 V2 coarse tiles complete", result.stdout)
            predictions = list(
                output_root.glob("v2_batch*/images/predictions_npy/*.npy")
            )
            self.assertEqual([path.name for path in predictions], ["tile-000.npy"])
            config = json.loads(
                (output_root / "hub-run-config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["maxTiles"], 1)
            self.assertEqual(
                config["selectedTiles"], ["v2_inputs/batch0/tile-000.png"]
            )


if __name__ == "__main__":
    unittest.main()
