"""Build a lunar_nac dataset for MMDE-Studio from existing reconstruction results.

Source data (read-only):
  - LROC NAC orthophoto/DTM triplets: /mnt/d/nac/official_rdr/expanded/{region}/
      NAC_DTM_{region}.TIF            float32 DTM, 2 m/px (height reference)
      NAC_DTM_{region}_*_2M.TIF       uint8 orthophoto, 2 m/px
  - Per-model predictions on the DTM grid (float32 GeoTIFF, nodata=-9999):
      /home/research/code/mono_depth_estimate/outputs/{model}/{region}_prediction.tif
    Models with a complete set of region predictions are included; partial
    tracks (dmb_depth_fold0, marigold_v2_input256, coarse32) are skipped.

Output layout (MMDE conventions, see backend/docs/API.md):
  {out}/mmde_test/lunar_nac/test_sequence/
      frames.jsonl                 one line per region ("frame")
      images/{region}.jpg          orthophoto downscaled to MAX_EDGE
      gt/{idx:06d}.npz             keys: uv (N,2) f32, depth (N,) f32
      sparse_controls/dtm/{idx:06d}.npz   keys: uv, depth, weight
  {out}/mmde_result/preds/lunar_nac/test_sequence/{model}/{idx:06d}.npy
  {out}/mmde_result/metrics/lunar_nac/test_sequence/{model}.json
  {out}/mmde_result/summary.json
  {out}/mmde_scene/lunar_nac/test_sequence/{model}/index.json + blobs

Depth convention: the orthophoto is a map, not a perspective image; each
region becomes one standalone frame with pseudo-pinhole K and identity
rotation. Depths are elevations in hectometers (hm = 100 m) relative to the
region's minimum valid DTM elevation, plus 0.5 hm, so values stay inside the
backend's GT validity window (MIN_DEPTH=0.1, MAX_DEPTH=200). Per-frame affine
alignment (a*pred+b against sampled DTM points) is applied by the backend at
serve time, exactly as for the vehicle datasets.

Scene blobs follow the documented format-2 layout: one group per region,
points on the region's geo offset (x/y in km from a reference origin, z in
hm), u16 quantization per group.

Usage:
  /home/zhy/miniconda3/envs/lunarecon/bin/python scripts/prepare_lunar_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio

NAC_ROOT = Path("/mnt/d/nac/official_rdr/expanded")
EVAL_OUTPUTS = Path("/home/research/code/mono_depth_estimate/outputs")
OUT = Path(__file__).resolve().parents[1] / "data"
DATASET = "lunar_nac"
SPLIT = "test_sequence"
MODELS = ["dav2_small", "dav2_small_input256", "marigold_v1_1_input256"]

MAX_EDGE = 1408          # viewer image long edge, px
PRED_NODATA = -9999
GT_MARGIN_HM = 0.5       # depths start this far above zero
SCENE_MAX_POINTS = 60_000

TEST_ROOT = OUT / "mmde_test" / DATASET / SPLIT
PRED_ROOT = OUT / "mmde_result" / "preds" / DATASET / SPLIT
METRICS_ROOT = OUT / "mmde_result" / "metrics" / DATASET / SPLIT
SCENE_ROOT = OUT / "mmde_scene" / DATASET / SPLIT


def log(msg: str) -> None:
    print(msg, flush=True)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), size,
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def fit_affine(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    A = np.stack([pred, np.ones(len(pred))], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, gt, rcond=None)
    # flat terrain can leave the pred variance ~0 and the least-squares slope
    # explodes; fall back to a constant estimate instead of insane alignments
    if not (np.isfinite(a) and np.isfinite(b)) or abs(a) > 1e6:
        a, b = 0.0, float(np.median(gt))
    return float(a), float(b)


def region_metrics(aligned: np.ndarray, gt: np.ndarray) -> dict:
    err = aligned - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.maximum(gt, 1e-3)
    abs_rel = float(np.mean(np.abs(err) / denom))
    ratio = np.maximum(aligned / np.maximum(gt, 1e-3),
                       gt / np.maximum(aligned, 1e-3))
    d1 = float(np.mean(ratio < 1.25))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "abs_rel": abs_rel, "d1": d1,
            "r2": r2, "n_pixels": int(len(gt))}


def write_scene_blob(path: Path, pos: np.ndarray, rgb: np.ndarray) -> None:
    """u32 n | u8 rgb[3n] | pad ((3n)&1) | u16 qx qy qz (scalar quant step)."""
    n = len(rgb)
    lo = pos.min(axis=0)
    step = float(max(pos.max(axis=0) - lo)) / 65535.0
    step = max(step, 1e-6)
    q = np.round((pos - lo) / step).clip(0, 65535).astype("<u2")
    buf = bytearray()
    buf += np.array([n], dtype="<u4").tobytes()
    buf += rgb.astype(np.uint8).tobytes()
    buf += b"\x00" * ((3 * n) & 1)
    buf += q[:, 0].tobytes() + q[:, 1].tobytes() + q[:, 2].tobytes()
    path.write_bytes(bytes(buf))


def main() -> int:
    regions = sorted(p for p in NAC_ROOT.iterdir() if p.is_dir())
    if not regions:
        log(f"no regions under {NAC_ROOT}")
        return 1
    # reference geo origin for scene world coordinates (first region corner)
    with rasterio.open(next(regions[0].glob("NAC_DTM_*.TIF")).parent
                       / f"NAC_DTM_{regions[0].name}.TIF") as r:
        e_ref, n_ref = r.transform.c, r.transform.f

    for d in (TEST_ROOT / "images", TEST_ROOT / "gt",
              TEST_ROOT / "sparse_controls" / "dtm"):
        d.mkdir(parents=True, exist_ok=True)

    frames = []
    summary: dict = {}
    scene_meta: dict[str, list] = {m: [] for m in MODELS}

    for idx, region in enumerate(regions):
        name = region.name
        dtm_path = region / f"NAC_DTM_{name}.TIF"
        img_path = next(region.glob("*_2M.TIF"))
        with rasterio.open(dtm_path) as dtm:
            elev = dtm.read(1, masked=True)
            valid_full = ~np.ma.getmaskarray(elev) & np.isfinite(elev.data)
            gt_full = elev.data
            transform = dtm.transform
            dtm_h, dtm_w = elev.shape
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            log(f"skip {name}: unreadable orthophoto")
            continue
        if img.shape != (dtm_h, dtm_w):
            # one region's orthophoto is a single row short of the DTM: crop
            # both to the common footprint instead of dropping the region
            ch, cw = min(dtm_h, img.shape[0]), min(dtm_w, img.shape[1])
            log(f"{name}: crop orthophoto {img.shape} / DTM "
                f"{(dtm_h, dtm_w)} -> {(ch, cw)}")
            img = img[:ch, :cw]
            gt_full, valid_full = gt_full[:ch, :cw], valid_full[:ch, :cw]
            dtm_h, dtm_w = ch, cw

        scale = MAX_EDGE / max(dtm_h, dtm_w)
        w, h = max(1, round(dtm_w * scale)), max(1, round(dtm_h * scale))
        img_ds = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(TEST_ROOT / "images" / f"{name}.jpg"), img_ds,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        # depth on the full grid first: elevations relative to the region's
        # minimum valid DTM elevation, in hm. NaN marks invalid; resizing
        # propagates NaN across the nodata boundary, which is then cut away
        # (a plain resize of raw DTM bleeds +-3.4e38 into valid pixels).
        elev_min = float(gt_full[valid_full].min())
        depth_full = np.where(
            valid_full,
            (gt_full - elev_min) / 100.0 + GT_MARGIN_HM,
            np.nan).astype(np.float32)
        depth = cv2.resize(depth_full, (w, h), interpolation=cv2.INTER_LINEAR)
        valid = np.isfinite(depth) \
            & resize_mask(valid_full, (w, h))
        depth = np.where(valid, depth, np.nan).astype(np.float32)

        # sparse GT / control samples on a regular grid (~150 per axis max)
        step = max(1, round(min(h, w) / 150))
        rows, cols = np.mgrid[step // 2:h:step, step // 2:w:step]
        m = valid[rows, cols]
        uv = np.stack([cols[m], rows[m]], axis=1).astype(np.float32)
        d = depth[rows, cols][m]
        if len(d) > 30_000:  # cap sparse-sample count for wire payloads
            keep = np.arange(0, len(d), int(np.ceil(len(d) / 30_000)))
            uv, d = uv[keep], d[keep]
        gt_npz = TEST_ROOT / "gt" / f"{idx:06d}.npz"
        np.savez_compressed(gt_npz, uv=uv, depth=d.astype(np.float32))
        np.savez_compressed(TEST_ROOT / "sparse_controls" / "dtm"
                            / f"{idx:06d}.npz", uv=uv,
                            depth=d.astype(np.float32),
                            weight=np.ones(len(d), np.float32))

        fx = fy = float(max(w, h))
        k = [[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]]
        # geo offset in hectometers puts regions side by side at the same
        # scale as depth (1 unit = 100 m) in Explorer registration and scenes
        tx, ty = (transform.c - e_ref) / 100.0, -(transform.f - n_ref) / 100.0
        frames.append({
            "seq_id": name, "frame_id": 0, "timestamp": float(idx),
            "camera": "nac_2m",
            "image_path": str(TEST_ROOT / "images" / f"{name}.jpg"),
            "K": k,
            "T_world_camera": [[1, 0, 0, tx], [0, 1, 0, ty],
                               [0, 0, 1, 0], [0, 0, 0, 1]],
            "depth_gt_path": str(gt_npz),
        })
        log(f"[{idx + 1}/{len(regions)}] {name}: image {w}x{h}, "
            f"{len(d)} gt/control points")

        # scene geometry shared by all models (region placement)
        px_hm = 2.0 / 100.0 * scale  # 2 m/px -> hm
        gx, gy = np.meshgrid(np.arange(w) * px_hm, np.arange(h) * px_hm)
        rgb_ds = cv2.cvtColor(img_ds, cv2.COLOR_GRAY2RGB).reshape(-1, 3)

        for model in MODELS:
            pred_path = EVAL_OUTPUTS / model / f"{name}_prediction.tif"
            model_dir = PRED_ROOT / model
            model_dir.mkdir(parents=True, exist_ok=True)
            with rasterio.open(pred_path) as src:
                pred = src.read(1)
            pvalid = resize_mask(pred != PRED_NODATA, (w, h)) & valid
            pred_ds = cv2.resize(np.where(pred != PRED_NODATA, pred, 0.0)
                                 .astype(np.float32), (w, h),
                                 interpolation=cv2.INTER_LINEAR)
            pred_ds = np.where(pvalid, pred_ds, np.nan).astype(np.float32)
            np.save(model_dir / f"{idx:06d}.npy", pred_ds)

            # oracle per-region affine fit + shape metrics (hm domain);
            # aligned depths are clipped to gt range +- 5x span so a wild
            # slope can not blow up scene extents or metric accumulations
            s = 4
            p_s, g_s = pred_ds[::s, ::s], depth[::s, ::s]
            ok = np.isfinite(p_s) & np.isfinite(g_s)
            a, b = fit_affine(p_s[ok], g_s[ok])
            aligned = (a * pred_ds + b).astype(np.float64)
            gfin = depth[np.isfinite(depth)]
            gspan = max(float(gfin.max() - gfin.min()), 1.0)
            aligned = np.clip(aligned,
                              float(gfin.min()) - 5 * gspan,
                              float(gfin.max()) + 5 * gspan)
            met = region_metrics(aligned[::s, ::s][ok],
                                 g_s[ok].astype(np.float64))
            met.update({"idx": idx, "seq_id": name, "frame_id": 0})
            scene_meta[model].append({
                "metrics": met, "aligned": aligned, "pvalid": pvalid,
                "gx": gx, "gy": gy, "tx": tx, "ty": ty, "rgb": rgb_ds,
                "w": w, "h": h,
            })

    with open(TEST_ROOT / "frames.jsonl", "w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")

    # metrics json + summary + scenes
    summary[DATASET + "/" + SPLIT] = {}
    for model in MODELS:
        per_frame = [m["metrics"] for m in scene_meta[model]]
        numeric = [k for k in per_frame[0]
                   if k not in ("idx", "seq_id", "frame_id")]
        aggregate = {k: float(np.mean([pf[k] for pf in per_frame]))
                     for k in numeric}
        doc = {"model": model, "dataset": DATASET, "split": SPLIT,
               "protocol": "region_oracle_aligned_hm",
               "aggregate": aggregate, "per_frame": per_frame}
        METRICS_ROOT.mkdir(parents=True, exist_ok=True)
        (METRICS_ROOT / f"{model}.json").write_text(
            json.dumps(doc, indent=2), encoding="utf-8")
        summary[DATASET + "/" + SPLIT][model] = aggregate

        scene_dir = SCENE_ROOT / model
        scene_dir.mkdir(parents=True, exist_ok=True)
        groups, cam_pos, cam_quat, cam_t, seqs = [], [], [], [], []
        n_points = 0
        all_lo = np.full(3, np.inf)
        all_hi = np.full(3, -np.inf)
        for idx, meta in enumerate(scene_meta[model]):
            aligned, pvalid = meta["aligned"], meta["pvalid"]
            stride = max(1, int(np.sqrt(meta["w"] * meta["h"]
                                        / SCENE_MAX_POINTS)) + 1)
            sub = pvalid[::stride, ::stride]
            gx = meta["gx"][::stride, ::stride][sub]
            gy = meta["gy"][::stride, ::stride][sub]
            gz = aligned[::stride, ::stride][sub]
            pos = np.stack([gx + meta["tx"], gy + meta["ty"], gz],
                           axis=1).astype(np.float64)
            rgb = meta["rgb"].reshape(meta["h"], meta["w"],
                                      3)[::stride, ::stride][sub]
            name = f"overview_s{idx}.bin"
            write_scene_blob(scene_dir / name, pos, rgb)
            lo, hi = pos.min(axis=0), pos.max(axis=0)
            all_lo, all_hi = np.minimum(all_lo, lo), np.maximum(all_hi, hi)
            groups.append({
                "seq_id": meta["metrics"]["seq_id"], "start": idx,
                "end": idx + 1, "n_points": int(len(rgb)),
                "bbox": {"min": lo.tolist(), "max": hi.tolist()},
                "origin": lo.tolist(),
                "quant_step": float(max(hi - lo)) / 65535.0,
                "overview": name, "overview_n": int(len(rgb)),
                "chunks": [],
            })
            cam_pos.append([float((meta["tx"] + gx.mean())),
                            float((meta["ty"] + gy.mean())),
                            float(gz.max() + 20.0)])
            cam_quat.append([1.0, 0.0, 0.0, 0.0])
            cam_t.append(float(idx))
            seqs.append({"seq_id": meta["metrics"]["seq_id"],
                         "start": idx, "end": idx + 1})
            n_points += len(rgb)
        index = {"format": 2, "dataset": DATASET, "split": SPLIT,
                 "model": model, "n_frames": len(frames), "voxel": 0.0,
                 "depth_scale": 0.01,
                 "n_points": n_points,
                 "bbox": {"min": all_lo.tolist(), "max": all_hi.tolist()},
                 "groups": groups, "cam_pos": cam_pos, "cam_quat": cam_quat,
                 "cam_t": cam_t, "seqs": seqs}
        (scene_dir / "index.json").write_text(json.dumps(index, indent=2),
                                              encoding="utf-8")
        log(f"scene {model}: {n_points} points, "
            f"bbox extent {(all_hi - all_lo).round(1).tolist()}")

    summary_path = OUT / "mmde_result" / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {len(frames)} frames, {len(MODELS)} models -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
