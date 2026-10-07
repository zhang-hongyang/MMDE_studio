#!/usr/bin/env python3
"""Linear-probe elevation readout from frozen SOMA features.

Protocol (mirrors the lunar_nac_eval benchmark, geographic split):
- train/val = supervised_splits.json fold 0 (18 / 2 regions)
- X: per-token encoder features (default last block, 768-d)
- y: DTM patch mean (m), 16x16 grid per tile
- global ridge probe, no per-test-tile fitting (that diagnostic is reported
  separately as oracle-affine)

Reports per val region: absolute MAE/RMSE, R^2, oracle-affine MAE/RMSE
(shape diagnostic), per-tile demeaned RMSE; constant baselines; DAV2
relative-depth single-feature control probe.

Run with .venv_lfm (needs numpy, rasterio for the DAV2 control).
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "outputs/lfm_probe"
FEAT = PROBE / "features"
LABEL = PROBE / "labels"
SPLITS = ROOT / "outputs/supervised_splits.json"
DATA = Path("/mnt/d/nac/official_rdr/expanded")
DAV2 = ROOT / "outputs/dav2_small_input256"
FOLD = 0
LAYER = -1          # encoder block output used as feature
RIDGE_LAM = 1e3
SUBSAMPLE = 2       # keep every n-th train token
SEED = 20260917


def load_split():
    s = json.loads(SPLITS.read_text())
    f = s["folds"][FOLD]
    return f["train"], f["val"]


def tile_rows():
    import csv
    return list(csv.DictReader((ROOT / "outputs/marigold_v2_input256/tiles_manifest.csv").open()))


def collect(rows, regions, rng=None, tile_frac=1.0):
    """Return X (n,768) float16, y (n,), tile_id (n,), region (n,)."""
    Xs, ys, tid, reg = [], [], [], []
    for r in rows:
        if r["region"] not in regions:
            continue
        if rng is not None and tile_frac < 1.0 and rng.random() > tile_frac:
            continue
        stem = r["file"].replace(".png", "")
        lf = FEAT / f"{stem}_f.npy"
        lb = LABEL / f"{r['region']}_{stem}_label.npy"
        if not (lf.exists() and lb.exists()):
            continue
        f = np.load(lf)[LAYER]            # (256,768)
        lbl, msk = np.load(lb)            # (16,16) each
        f = f.reshape(16, 16, 768)
        valid = msk.astype(bool).ravel()
        Xs.append(f.reshape(-1, 768)[valid])
        ys.append(lbl.reshape(-1)[valid])
        tid.append(np.full(valid.sum(), hash(stem) % (2**31)))
        reg.extend([r["region"]] * int(valid.sum()))
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(tid), np.array(reg))


def ridge_fit(X, y, lam):
    xm = X.mean(0)
    ym = y.mean()
    Xc = X - xm
    A = (Xc.T @ Xc).astype(np.float64) + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, (Xc.T @ (y - ym)).astype(np.float64))
    return xm, ym, w


def ridge_pred(model, X):
    xm, ym, w = model
    return ym + (X.astype(np.float64) - xm) @ w


def affine_align(y, p):
    A = np.stack([p, np.ones_like(p)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef, coef


def region_metrics(y, p, tid, name):
    out = {"region": name, "n": int(len(y))}
    out["mae_abs"] = float(np.abs(y - p).mean())
    out["rmse_abs"] = float(np.sqrt(((y - p) ** 2).mean()))
    ss = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out["r2_abs"] = float(ss)
    pa, (s, o) = affine_align(y, p)
    out["oracle_mae"] = float(np.abs(y - pa).mean())
    out["oracle_rmse"] = float(np.sqrt(((y - pa) ** 2).mean()))
    # per-tile demeaned shape error
    df_y, df_p = y - p, np.zeros_like(y)
    for t in np.unique(tid):
        m = tid == t
        df_p[m] = p[m] - p[m].mean()
        df_y[m] = y[m] - y[m].mean()
    out["shape_rmse"] = float(np.sqrt(((df_y - df_p) ** 2).mean()))
    out["constant_mae"] = float(np.abs(y - y.mean()).mean())  # oracle constant
    return out


def dav2_features(rows, regions):
    """DAV2 relative depth (16x16) as a single feature + labels."""
    Xs, ys, tid, reg = [], [], [], []
    for r in rows:
        if r["region"] not in regions:
            continue
        stem = r["file"].replace(".png", "")
        lb = LABEL / f"{r['region']}_{stem}_label.npy"
        pred = DAV2 / f"{r['region']}_prediction.tif"
        if not (lb.exists() and pred.exists()):
            continue
        x, y, w, h = (int(r[k]) for k in ("x", "y", "width", "height"))
        with rasterio.open(pred) as ds:
            a = ds.read(1, window=Window(x, y, w, h))
        a = np.where(np.isfinite(a), a, 0.0)
        if w < 512 or h < 512:
            pad = np.zeros((512, 512), np.float32)
            pad[:h, :w] = a
            a = pad
        d16 = a.reshape(16, 32, 16, 32).transpose(0, 2, 1, 3).mean(axis=(2, 3)).reshape(-1)
        lbl, msk = np.load(lb)
        valid = msk.astype(bool).reshape(-1)
        Xs.append(d16[valid][:, None])
        ys.append(lbl.reshape(-1)[valid])
        tid.append(np.full(valid.sum(), hash(stem) % (2**31)))
        reg.extend([r["region"]] * int(valid.sum()))
    if not Xs:
        return None
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(tid), np.array(reg))


def main():
    rng = np.random.default_rng(SEED)
    train_r, val_r = load_split()
    rows = tile_rows()
    print("train regions:", len(train_r), "val:", val_r)

    # subsample train tiles up front to bound RAM (15 GB machine)
    Xtr, ytr, tidtr, regtr = collect(rows, set(train_r), rng=rng, tile_frac=0.6)
    print("train tokens:", Xtr.shape)
    keep = rng.choice(len(Xtr), len(Xtr) // SUBSAMPLE, replace=False)
    model = ridge_fit(Xtr[keep].astype(np.float32), ytr[keep], RIDGE_LAM)
    train_mean = float(ytr[keep].mean())
    del Xtr

    results = {"fold": FOLD, "layer": LAYER, "lam": RIDGE_LAM,
               "train_regions": train_r, "val_regions": val_r, "soma": [], "dav2": [], "constant": []}
    for vr in val_r:
        Xv, yv, tidv, regv = collect(rows, {vr})
        p = ridge_pred(model, Xv)
        results["soma"].append(region_metrics(yv, p, tidv, vr))
        results["constant"].append(region_metrics(yv, np.full_like(yv, train_mean), tidv, vr))
        d = dav2_features(rows, {vr})
        if d is not None:
            Xd, yd, tidd, _ = d
            Xtr_d, ytr_d, _, _ = dav2_features(rows, set(train_r))
            m2 = ridge_fit(Xtr_d.astype(np.float32), ytr_d, 1.0)
            pd = ridge_pred(m2, Xd)
            results["dav2"].append(region_metrics(yd, pd, tidd, vr))
        print(vr, "soma rmse_abs",
              f"{results['soma'][-1]['rmse_abs']:.2f}",
              "oracle_rmse", f"{results['soma'][-1]['oracle_rmse']:.2f}", flush=True)

    out = PROBE / f"probe_results_fold{FOLD}.json"
    out.write_text(json.dumps(results, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
