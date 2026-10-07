"""Run the official V2 infer entry point on one 256 px pilot and log peak VRAM."""

import json
import subprocess
import time
from pathlib import Path


BASE = Path(__file__).parent
REPO = Path("/home/research/code/MMDE_studio/algorithms/marigold-v2")
PYTHON = Path("/home/research/code/MMDE_studio/algorithms/.venv_marigold/bin/python")
OUT = BASE / "outputs/v2_pilot_256"


def used_memory_mib():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.splitlines()[0].strip())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), str(REPO / "scripts/infer.py"),
               "--image_dir", str(BASE / "pilot_inputs"),
               "--output_dir", str(OUT / "inference"),
               "--width", "256", "--height", "256"]
    start = time.monotonic()
    baseline = used_memory_mib()
    peak = baseline
    log_path = OUT / "infer.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=log,
                                   stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                peak = max(peak, used_memory_mib())
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            time.sleep(0.5)
    result = {"command": command, "returncode": process.returncode,
              "elapsed_seconds": round(time.monotonic() - start, 2),
              "baseline_gpu_memory_mib": baseline,
              "peak_gpu_memory_mib": peak,
              "log": str(log_path)}
    (OUT / "memory_test.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
