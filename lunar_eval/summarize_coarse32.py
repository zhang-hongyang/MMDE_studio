"""Region-level paired uncertainty for the separate 32 m full-coverage track."""

import csv
import json
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).parent / "outputs/coarse32/region_metrics.csv"
OUT = Path("/mnt/d/nac/evaluation_20260914/coarse32_summary.json")


def bootstrap_ci(values, rng, reps=20000):
    indices = rng.integers(len(values), size=(reps, len(values)))
    sample = np.median(values[indices], axis=1)
    return np.quantile(sample, (0.025, 0.975)).tolist()


def main():
    with SOURCE.open(newline="") as file:
        records = {(row["region"], row["method"]): row
                   for row in csv.DictReader(file)}
    regions = sorted({region for region, _ in records})
    methods = ("dav2", "marigold_v1", "marigold_v2")
    if len(regions) != 20 or len(records) != 60:
        raise ValueError("Expected 20 paired regions and 3 methods")
    values = {method: np.array([float(records[(region, method)]["oracle_mae_m"])
                                for region in regions]) for method in methods}
    values["constant"] = np.array([
        float(records[(region, "dav2")]["oracle_constant_mae_m"])
        for region in regions])
    rng = np.random.default_rng(20260914)
    comparisons = {}
    for treatment, control in (("dav2", "constant"),
                               ("marigold_v1", "constant"),
                               ("marigold_v2", "constant"),
                               ("marigold_v2", "dav2"),
                               ("marigold_v2", "marigold_v1")):
        improvement = (values[control] - values[treatment]) / values[control]
        comparisons[f"{treatment}_vs_{control}"] = {
            "wins_of_20": int(np.sum(values[treatment] < values[control])),
            "median_relative_mae_reduction": float(np.median(improvement)),
            "bootstrap_95ci": bootstrap_ci(improvement, rng),
        }
    result = {
        "unit": "region", "n_regions": 20,
        "reference_resolution_m": 32,
        "valid_pixels": sum(int(records[(r, "dav2")]["pixels"])
                            for r in regions),
        "metric": "test-region oracle affine-aligned MAE; not absolute elevation",
        "median_region_mae_m": {name: float(np.median(value))
                                for name, value in values.items()},
        "median_oracle_r2": {
            method: float(np.median([
                float(records[(region, method)]["oracle_r2"])
                for region in regions])) for method in methods
        },
        "comparisons": comparisons,
        "bootstrap_seed": 20260914, "bootstrap_resamples": 20000,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
