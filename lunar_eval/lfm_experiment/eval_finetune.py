#!/usr/bin/env python3
"""Evaluate a trained NAC->DTM checkpoint on val regions with probe-comparable metrics.

Metrics per region (same protocol as run_probe.py):
  - mae_abs       : MAE vs raw DTM labels (no per-tile fitting)
  - oracle_mae / oracle_rmse : single affine a*y+b fit per REGION on all pixels
  - shape_rmse    : RMSE after per-TILE demeaning (high-freq shape fidelity)
Baselines reported for comparison: constant per-region mean, step-1 SOMA probe.

Usage:
  .venv_lfm/bin/python eval_finetune.py --ckpt tb_logs/nac_dtm/checkpoints/ni_lfm_lora/best-*.ckpt \
      --split val --tag lora
Writes outputs/lfm_probe/finetune_{tag}_metrics.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent / "ni_lfm_repo"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/nac_dtm"
MEAN, STD = 0.053995178164645, 0.0374660347301751
# labels are z-scored with train-global stats (normalize_labels.py)
LS = json.loads((DATA / "label_stats.json").read_text())
LBL_MEAN, LBL_STD = LS["mean"], LS["std"]


def load_model(ckpt_path, device):
    sys_path_fix = str(REPO)
    import sys
    if sys_path_fix not in sys.path:
        sys.path.insert(0, sys_path_fix)
    from terratorch_integration import LunarPixelwiseRegressionTask  # noqa
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = {k: v for k, v in ckpt["hyper_parameters"].items()
               if not k.startswith("_")}
    task = LunarPixelwiseRegressionTask(**hparams)
    task.load_state_dict(ckpt["state_dict"])
    task.to(device).eval()
    for p in task.parameters():
        p.requires_grad_(False)
    return task


def read_label(path):
    """Return (z-scored label array, valid mask)."""
    with rasterio.open(path) as d:
        a = d.read(1, masked=True)
    return a.data.astype(np.float32), ~np.ma.getmaskarray(a)


def region_of(stem):
    # stems look like A17SIVB_y00000_x00000_w512_h512
    return stem.rsplit("_y", 1)[0]


@torch.no_grad()
def predict(task, stems, device, batch=8):
    outs = {}
    for i in range(0, len(stems), batch):
        chunk = stems[i:i + batch]
        xs = []
        for s in chunk:
            with rasterio.open(DATA / f"{s}_NAC.tif") as d:
                a = d.read(1).astype(np.float32)
            xs.append(torch.from_numpy((a - MEAN) / STD)[None])
        x = torch.stack(xs).to(device)
        out = task({"nac": x})
        y_pred = getattr(out, "pred", None) or out.output
        if isinstance(y_pred, dict):
            y_pred = next(iter(y_pred.values()))
        y = y_pred.squeeze(1).float().cpu().numpy() * LBL_STD + LBL_MEAN  # un-normalize to meters
        for s, p in zip(chunk, y):
            outs[s] = p
    return outs


def affine_fit(pred, gt):
    A = np.stack([pred, np.ones_like(pred)], 1)
    coef, *_ = np.linalg.lstsq(A, gt, rcond=None)
    return coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--tag", default="lora")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = load_model(args.ckpt, device)

    stems = [Path(l).stem for l in (DATA / "splits" / f"{args.split}.txt").read_text().split()]
    preds = predict(task, stems, device)

    regions = {}
    for s in stems:
        regions.setdefault(region_of(s), []).append(s)

    rows = []
    for reg, ss in sorted(regions.items()):
        P, G, M = [], [], []
        for s in ss:
            g, m = read_label(DATA / f"{s}_DTM.tif")
            g = g * LBL_STD + LBL_MEAN  # to meters
            p = preds[s]
            P.append(p[m]); G.append(g[m]); M.append(m)
        P = np.concatenate(P); G = np.concatenate(G)
        # constant baseline (train-free): per-region mean of labels
        const = np.full_like(G, G.mean())
        a, b = affine_fit(P, G)
        Pa = a * P + b
        mae_abs = np.abs(P - G).mean()
        oracle_mae = np.abs(Pa - G).mean()
        oracle_rmse = np.sqrt(((Pa - G) ** 2).mean())
        const_mae = np.abs(const - G).mean()
        # shape_rmse: per-tile demeaned
        sr = []
        for s in ss:
            g, m = read_label(DATA / f"{s}_DTM.tif")
            g = g * LBL_STD + LBL_MEAN  # to meters
            p, gg = preds[s][m], g[m]
            sr.append(((p - p.mean()) - (gg - gg.mean())) ** 2)
        shape_rmse = np.sqrt(np.concatenate(sr).mean())
        rows.append(dict(region=reg, n_tiles=len(ss), mae_abs=mae_abs,
                         oracle_mae=oracle_mae, oracle_rmse=oracle_rmse,
                         const_mae=const_mae, shape_rmse=shape_rmse,
                         affine_a=a, affine_b=b))
        print(f"{reg:14s} n={len(ss):4d}  rawMAE={mae_abs:7.1f}  oracleMAE={oracle_mae:7.1f} "
              f"oracleRMSE={oracle_rmse:7.1f}  constMAE={const_mae:7.1f}  shapeRMSE={shape_rmse:7.1f}  "
              f"a={a:.3f} b={b:.0f}")

    out = ROOT / "outputs/lfm_probe"
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with (out / f"finetune_{args.tag}_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    # save raw preds for figure-making
    np.savez(out / f"finetune_{args.tag}_preds.npz",
             **{s: preds[s] for s in stems})
    print("saved", out / f"finetune_{args.tag}_metrics.csv")


if __name__ == "__main__":
    main()
