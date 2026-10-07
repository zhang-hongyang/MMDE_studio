#!/usr/bin/env python3
"""Sweep probe over folds x layers x ridge lambda; cache features on disk.

Outputs a compact CSV: fold, layer, lam, region, method(soma/dav2/constant),
n, mae_abs, rmse_abs, oracle_mae, oracle_rmse, shape_rmse, r2_abs.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_probe import (FEAT, LABEL, SPLITS, collect, ridge_fit, ridge_pred,
                       region_metrics, dav2_features, tile_rows)  # noqa: E402

PROBE = ROOT / "outputs/lfm_probe"
OUT_CSV = PROBE / "sweep_results.csv"
SEED = 20260917
LAYERS = [3, 6, 9, 11]
LAMS = [1e2, 1e3, 1e4, 1e5]


def set_layer(idx):
    import run_probe
    run_probe.LAYER = idx


def main():
    rng = np.random.default_rng(SEED)
    splits = json.loads(SPLITS.read_text())
    rows = tile_rows()
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "layer", "lam", "region", "method", "n",
                    "mae_abs", "rmse_abs", "oracle_mae", "oracle_rmse",
                    "shape_rmse", "r2_abs"])
        for fold in splits["folds"]:
            train_r, val_r = fold["train"], fold["val"]
            Xd_tr, yd_tr, _, _ = dav2_features(rows, set(train_r))
            m2 = ridge_fit(Xd_tr.astype(np.float32), yd_tr, 1.0)
            del Xd_tr, yd_tr
            for layer in LAYERS:
                set_layer(layer)
                Xtr, ytr, _, _ = collect(rows, set(train_r), rng=rng, tile_frac=0.6)
                if layer == LAYERS[0]:
                    ytr_k0 = ytr[rng.choice(len(ytr), len(ytr) // 2, replace=False)]
                    train_mean = float(ytr_k0.mean())
                keep = rng.choice(len(Xtr), len(Xtr) // 2, replace=False)
                ytr_k = ytr[keep]
                Xl = Xtr[keep].astype(np.float32)
                del Xtr
                for lam in LAMS:
                    model = ridge_fit(Xl, ytr_k, lam)
                    for vr in val_r:
                        Xv, yv, tidv, _ = collect(rows, {vr})
                        p = ridge_pred(model, Xv)
                        r = region_metrics(yv, p, tidv, vr)
                        w.writerow([fold["fold"], layer, lam, vr, "soma", r["n"],
                                    r["mae_abs"], r["rmse_abs"], r["oracle_mae"],
                                    r["oracle_rmse"], r["shape_rmse"], r["r2_abs"]])
                        if layer == LAYERS[0] and lam == LAMS[0]:
                            rc = region_metrics(yv, np.full_like(yv, train_mean), tidv, vr)
                            w.writerow([fold["fold"], layer, lam, vr, "constant", rc["n"],
                                        rc["mae_abs"], rc["rmse_abs"], rc["oracle_mae"],
                                        rc["oracle_rmse"], rc["shape_rmse"], rc["r2_abs"]])
                            Xd, yd, tidd, _ = dav2_features(rows, {vr})
                            pd_ = ridge_pred(m2, Xd)
                            rd = region_metrics(yd, pd_, tidd, vr)
                            w.writerow([fold["fold"], layer, lam, vr, "dav2", rd["n"],
                                        rd["mae_abs"], rd["rmse_abs"], rd["oracle_mae"],
                                        rd["oracle_rmse"], rd["shape_rmse"], rd["r2_abs"]])
                        fh.flush()
                    print(f"fold {fold['fold']} layer {layer} lam {lam:.0e} {vr} "
                          f"oracle_mae {r['oracle_mae']:.2f}", flush=True)
    print("saved:", OUT_CSV)


if __name__ == "__main__":
    main()
