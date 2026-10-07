"""Predeclared region-grouped 5-fold split for supervised lunar depth training.

Sampling unit is the REGION, never the tile (README protocol). Fixed seed,
committed before any supervised model is trained or selected on. Every one of
the 20 regions appears in exactly one test fold and, across folds, is eligible
for held-out prediction exactly once.
"""

import json
import random
from pathlib import Path

ROOT = Path("/mnt/d/nac/official_rdr/expanded")
OUT = Path(__file__).parent / "outputs/supervised_splits.json"
SEED = 20260913
N_FOLDS = 5
N_VAL = 2


def main():
    regions = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    assert len(regions) == 20, regions

    # Deterministic shuffle, independent of PYTHONHASHSEED.
    rng = random.Random(SEED)
    order = regions[:]
    rng.shuffle(order)

    n_test = len(order) // N_FOLDS  # 4
    folds = []
    for i in range(N_FOLDS):
        test = sorted(order[i * n_test:(i + 1) * n_test])
        remaining = [r for r in order if r not in test]
        # Validation rotates through the remaining pool so that across folds
        # every non-test region is used for validation the same number of
        # times where possible; val regions never overlap that fold's test.
        val_pool = remaining[:]
        rng_val = random.Random(SEED + i)
        rng_val.shuffle(val_pool)
        val = sorted(val_pool[:N_VAL])
        train = sorted(r for r in remaining if r not in val)
        folds.append({"fold": i, "train": train, "val": val, "test": test})

    # Sanity: union of test folds covers all regions exactly once.
    seen = [r for f in folds for r in f["test"]]
    assert sorted(seen) == regions, "test folds must cover each region once"
    for f in folds:
        assert not (set(f["train"]) & set(f["val"]))
        assert not (set(f["train"]) & set(f["test"]))
        assert not (set(f["val"]) & set(f["test"]))

    payload = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_val_regions_per_fold": N_VAL,
        "sampling_unit": "region",
        "regions": regions,
        "folds": folds,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    for f in folds:
        print(f"fold {f['fold']}: train={len(f['train'])} val={f['val']} test={f['test']}")


if __name__ == "__main__":
    main()
