"""Region-level paired summaries with bootstrap CIs, never pixel-level CI."""

import csv
import json
from pathlib import Path

import numpy as np


BASE = Path(__file__).parent / "outputs"
OUT = Path("/mnt/d/nac/evaluation_20260914/summary_input256.json")


def rows(path):
    with path.open(newline="") as file:
        return {row["region"]: row for row in csv.DictReader(file)}


def bootstrap_median(values, rng, reps=20000):
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    return np.quantile(np.median(values[indices], axis=1), [0.025, 0.975]).tolist()


def main():
    baseline = rows(BASE / "dav2_small/constant_metrics.csv")
    dav2 = rows(BASE / "dav2_small_input256/region_metrics.csv")
    v1 = rows(BASE / "marigold_v1_1_input256/region_metrics.csv")
    regions = sorted(set(baseline) & set(dav2) & set(v1))
    if len(regions) != 20:
        raise ValueError(f"Expected 20 paired regions, found {len(regions)}")
    constant_mae = np.array([float(baseline[r]["oracle_constant_mae_m"])
                             for r in regions])
    dav2_mae = np.array([float(dav2[r]["oracle_mae_m"]) for r in regions])
    v1_mae = np.array([float(v1[r]["oracle_mae_m"]) for r in regions])
    rng = np.random.default_rng(20260914)
    comparison = {}
    for name, treatment, control in (
        ("dav2_vs_constant", dav2_mae, constant_mae),
        ("marigold_v1_vs_constant", v1_mae, constant_mae),
        ("marigold_v1_vs_dav2", v1_mae, dav2_mae),
    ):
        gain = (control - treatment) / control
        comparison[name] = {
            "wins_of_20": int(np.sum(treatment < control)),
            "median_relative_mae_reduction": float(np.median(gain)),
            "median_relative_mae_reduction_bootstrap_95ci":
                bootstrap_median(gain, rng),
            "median_paired_mae_reduction_m": float(np.median(control - treatment)),
        }
    result = {
        "unit": "region",
        "n_regions": 20,
        "valid_dtm_pixels": sum(int(dav2[r]["pixels"]) for r in regions),
        "input": "512x512 2-m ortho tile resized to 256x256 for both models",
        "metric": "test-region oracle affine-aligned MAE; not absolute elevation",
        "bootstrap_seed": 20260914,
        "bootstrap_resamples": 20000,
        "median_region_mae_m": {
            "constant": float(np.median(constant_mae)),
            "dav2_256": float(np.median(dav2_mae)),
            "marigold_v1_256": float(np.median(v1_mae)),
        },
        "comparisons": comparison,
        "regions": regions,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
