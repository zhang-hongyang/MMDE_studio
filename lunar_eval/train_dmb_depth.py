"""Train the DMBNet-style lunar depth model on one fold.

Fold membership comes from the predeclared outputs/supervised_splits.json.
Depth target is the metric DTM; the region-level affine diagnostic is applied
at evaluation time (eval_dmb_depth.py), never during training or selection.

Loss = masked Huber(depth) + lambda_grad * masked gradient-L1 + lambda_b *
masked boundary BCE. Validation selects on the REGION-LEVEL oracle-affine MAE
(same protocol as the test evaluation, on held-out val regions), because the
deployed comparison is a single affine fit per region: per-tile-normalized
targets cap even a perfect shape predictor at ~constant MAE (verified
numerically), so the target is the raw metric DTM in a continuous frame.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dmb_depth_model import DMBDepth, gradient_loss

DATA = Path(os.environ.get("LUNAR_DATA", "/mnt/d/nac/supervised_v1"))
SPLITS = Path(__file__).parent / "outputs/supervised_splits.json"
OUT_ROOT = Path(__file__).parent / "outputs"


def boundary_target_numpy(dtm, mask, top_frac=0.15):
    """Top-`top_frac`-gradient valid pixels as a float boundary map. Cached.

    NoData sentinels in the raw DTM would overflow float32 gradients, so the
    masked-out value is excluded before differencing and a pixel is a boundary
    candidate only if it and its differencing neighbours are all valid.
    """
    out = np.zeros_like(dtm, dtype=np.float32)
    if mask.sum() < 100:
        return out
    d = np.where(mask, dtm, np.nan).astype(np.float64)
    gx = np.full_like(d, np.nan)
    gy = np.full_like(d, np.nan)
    gx[:, 1:-1] = (d[:, 2:] - d[:, :-2]) / 2
    gy[1:-1, :] = (d[2:, :] - d[:-2, :]) / 2
    okx = mask & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    oky = mask & np.roll(mask, 1, 0) & np.roll(mask, -1, 0)
    grad = np.hypot(np.where(okx, gx, 0.0), np.where(oky, gy, 0.0))
    cand = mask & okx & oky
    k = min(max(int(top_frac * mask.sum()), 1), int((grad[cand] > 0).sum()))
    if k == 0:
        return out
    thresh = np.partition(grad[cand], -k)[-k]
    out[(grad >= thresh) & cand & (grad > 0)] = 1.0
    return out


class TileDataset(Dataset):
    def __init__(self, regions):
        self.entries = []
        for region in regions:
            for path in sorted((DATA / region / "tiles").glob("*_img.npy")):
                key = path.name.removesuffix("_img.npy")
                self.entries.append((region, key))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        region, key = self.entries[idx]
        base = DATA / region / "tiles" / key
        img = np.load(base.parent / f"{key}_img.npy")
        dtm = np.load(base.parent / f"{key}_dtm.npy")
        mask = np.load(base.parent / f"{key}_mask.npy")
        bnd_path = base.parent / f"{key}_bnd.npy"
        if bnd_path.exists():
            bnd = np.load(bnd_path)
        else:
            bnd = boundary_target_numpy(dtm, mask)
            np.save(bnd_path, bnd)
        dino = np.load(DATA / "dinov3_s" / f"{region}_{key}_dino.npy")
        # Pad partial edge tiles to the full 512 square (mask stays invalid).
        h, w = img.shape
        ph, pw = 512 - h, 512 - w
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw)), mode="edge")
            dtm = np.pad(dtm, ((0, ph), (0, pw)), mode="edge")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant")
            bnd = np.pad(bnd, ((0, ph), (0, pw)), mode="constant")
        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        # Per-tile standardization of the (grayscale) orthophoto.
        if mask.any():
            v = img_t[0][torch.from_numpy(mask)]
            mu, sd = v.mean(), v.std().clamp_min(1e-6)
            img_t = (img_t - mu) / sd
        dtm_t = torch.from_numpy(dtm).float().unsqueeze(0)
        # Raw metric target in a continuous frame (see module docstring).
        # std placeholder keeps the collate shape; unused for raw targets.
        return (img_t, dtm_t, torch.from_numpy(mask).unsqueeze(0),
                torch.from_numpy(dino).float(), torch.from_numpy(bnd).unsqueeze(0),
                torch.tensor([1.0], dtype=torch.float32))


def augment(img, dtm, mask, dino, bnd):
    """Random dihedral transform (rotations + flips), applied consistently.

    The top-k boundary target commutes with any pixel permutation, so the
    cached boundary map is transformed with the same operator.
    """
    k = int(torch.randint(0, 4, (1,)))
    flip = bool(torch.randint(0, 2, (1,)))
    def t(x):
        x = torch.rot90(x, k, dims=(-2, -1))
        return torch.flip(x, dims=(-1,)) if flip else x
    return t(img), t(dtm), t(mask), t(dino), t(bnd)


def masked_huber(pred, target, mask, beta=1.0):
    err = pred - target
    abs_err = err.abs()
    quad = torch.clamp(abs_err, max=beta)
    loss = 0.5 * (quad ** 2) / beta + (abs_err - quad)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def region_oracle_mae(items):
    """One least-squares affine fit per region; returns MAE in meters.

    `items` is a list of (pred, target, mask) tiles for a single region.
    """
    ps, ts = [], []
    for pred, target, mask in items:
        ps.append(pred[mask].double())
        ts.append(target[mask].double())
    x = torch.cat(ps).numpy()
    y = torch.cat(ts).numpy()
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx, sxy = x @ x, x @ y
    denom = sxx - sx * sx / n
    scale = (sxy - sx * sy / n) / denom if denom > 0 else 0.0
    offset = (sy - scale * sx) / n
    return float(np.abs(scale * x + offset - y).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="4 fits the 12 GB card; 8 exceeds physical VRAM "
                             "and spills to shared memory (~10x slowdown)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-grad", type=float, default=1.0)
    parser.add_argument("--lambda-b", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    splits = json.loads(SPLITS.read_text())
    fold = next(f for f in splits["folds"] if f["fold"] == args.fold)
    out_dir = OUT_ROOT / f"dmb_depth_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TileDataset(fold["train"])
    val_ds = TileDataset(fold["val"])
    print(f"fold {args.fold}: train={len(train_ds)} tiles from {len(fold['train'])} regions, "
          f"val={len(val_ds)} tiles from {len(fold['val'])} regions, test={fold['test']}",
          flush=True)

    g = torch.Generator().manual_seed(args.seed)
    nw = min(args.workers, 8)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, generator=g, drop_last=True,
                              pin_memory=True, persistent_workers=nw > 0,
                              prefetch_factor=4 if nw > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(nw // 2, 1), pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)

    model = DMBDepth().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(train_loader))
    scaler = torch.cuda.amp.GradScaler()

    best_val = math.inf
    bad_epochs = 0
    log_path = out_dir / "train_log.csv"
    with log_path.open("w") as lf:
        lf.write("epoch,train_loss,val_mae,val_loss,lr,minutes\n")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        nb = 0
        for img, dtm, mask, dino, bnd, _std in train_loader:
            img, dtm, mask, dino, bnd = (img.cuda(), dtm.cuda(), mask.cuda(),
                                         dino.cuda(), bnd.cuda())
            img, dtm, mask, dino, bnd = augment(img, dtm, mask, dino, bnd)
            stages = [dino[:, i].contiguous() for i in range(4)]

            with torch.autocast("cuda", dtype=torch.float16):
                pred, blogits = model(img, stages)
                m = mask.float()
                loss_d = masked_huber(pred, dtm, m)
                loss_g = gradient_loss(pred, dtm, mask)
                loss_b = F.binary_cross_entropy_with_logits(blogits, bnd)
            loss = loss_d + args.lambda_grad * loss_g + args.lambda_b * loss_b

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item(); nb += 1
        train_loss = run / max(nb, 1)

        model.eval()
        val_items = {}
        val_losses = []
        with torch.inference_mode():
            pos = 0
            for img, dtm, mask, dino, bnd, _std in val_loader:
                img, dtm, mask, dino, bnd = (img.cuda(), dtm.cuda(), mask.cuda(),
                                             dino.cuda(), bnd.cuda())
                stages = [dino[:, i].contiguous().float() for i in range(4)]
                with torch.autocast("cuda", dtype=torch.float16):
                    pred, blogits = model(img, stages)
                    m = mask.float()
                    loss = (masked_huber(pred, dtm, m)
                            + args.lambda_grad * gradient_loss(pred, dtm, mask)
                            + args.lambda_b * F.binary_cross_entropy_with_logits(blogits, bnd))
                val_losses.append(loss.item())
                for b in range(img.shape[0]):
                    region = val_ds.entries[pos][0]
                    val_items.setdefault(region, []).append(
                        (pred[b, 0].float().cpu(), dtm[b, 0].float().cpu(),
                         mask[b, 0].cpu()))
                    pos += 1
        per_region = {r: region_oracle_mae(v) for r, v in val_items.items()}
        val_mae = float(np.mean(list(per_region.values())))
        val_loss = float(np.mean(val_losses)) if val_losses else math.inf

        improved = val_mae < best_val
        if improved:
            best_val = val_mae
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_mae": val_mae, "args": vars(args)},
                       out_dir / "checkpoint_best.pt")
        else:
            bad_epochs += 1
        print(f"epoch {epoch:3d} train={train_loss:.2f} val_mae={val_mae:.2f} "
              f"({', '.join(f'{r}={v:.1f}' for r, v in per_region.items())}) "
              f"val_loss={val_loss:.2f} lr={sched.get_last_lr()[0]:.2e} "
              f"{'*' if improved else ''} ({time.time()-t0:.0f}s)", flush=True)
        with log_path.open("a") as lf:
            lf.write(f"{epoch},{train_loss:.4f},{val_mae:.4f},{val_loss:.4f},"
                     f"{sched.get_last_lr()[0]:.6e},{(time.time()-t0)/60:.1f}\n")
        if bad_epochs >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    print(f"best val_mae={best_val:.3f}; checkpoint at {out_dir/'checkpoint_best.pt'}")


if __name__ == "__main__":
    main()
