# Lunar NAC terrain evaluation (20-region batch)

This benchmark uses the 20 official LROC NAC orthophoto/DTM triplets in
`/mnt/d/nac/official_rdr/expanded`. The DTM is 2 m/pixel and is the only
height reference. The 0.5–0.7 m orthophoto must not be scored against an
upsampled DTM as if that were high-resolution ground truth.

## Comparison set

1. Constant/planar elevation: no-image sanity-check lower bound.
2. Depth Anything V2 Small, zero-shot: runnable general-image monocular model.
3. Marigold V1.1 depth, zero-shot: diffusion-depth predecessor, evaluated.
4. Lunar Foundation Model frozen encoder + identical lightweight dense head:
   **supervised lunar-domain adaptation**, not a zero-shot baseline. Train on
   whole regions, select hyperparameters on held-out regions, test on untouched
   regions. Report separately from zero-shot methods.
5. Marigold V2 Log-stage2: target baseline, evaluated at 256 square input on
   a separate 32 m full-coverage track. Peak measured GPU use was 11,971 MiB
   on this 12 GB machine; the full 2 m tile track remains unevaluated for V2.

## Protocol

- Sampling unit is the **region**, not the tile. Keep every tile from a region
  in one split. Use the same DTM-valid footprint and 2 m reference grid for
  every method. Prefer same-resolution 2 m orthophoto for the main comparison;
  assess native high-resolution imagery separately.
- A predeclared 5-fold, region-grouped test gives every region one held-out
  prediction. For supervised models, further separate validation regions
  inside each training fold. For zero-shot models, no fitting on the test DTM.
- Monocular relative depth is not lunar metric elevation. Report shape metrics
  after **one affine fit per region** only as an oracle-aligned diagnostic;
  report raw metric MAE/RMSE only for a method that supplies metric elevation
  with a calibration fixed without test labels. Do not fit an affine transform
  per test tile and call the result reconstructive accuracy.
- Current shape metrics: region-level oracle-aligned MAE, RMSE, and (for the
  32 m track) R². Gradient and slope-angle error remain planned secondary
  metrics, not reported as completed. Include coverage, valid pixel count,
  and per-region results. Pair
  methods by region; summarize differences and bootstrap CIs resampling
  **regions**, not pixels or tiles.
- The LROC stereo DTM is a reference with nonzero measurement uncertainty, not
  perfect truth. Exclude NoData and obvious artifact regions; preserve their
  masks. A 2 m DTM cannot verify sub-2 m terrain detail.
- Audit overlap between test regions and any lunar-model pretraining data;
  otherwise label lunar-model results as in-domain rather than unseen-terrain
  generalization. For photometric/SfS methods, illumination and image geometry
  metadata are prerequisites and need a separate, fair evaluation track.

## Completed first-pass evaluation (2026-09-13)

Depth Anything V2 Small zero-shot and the oracle constant-elevation check were
evaluated on all 20 regions, using the same 758,340,549 valid 2 m DTM pixels.
DAV2 was run as nonoverlapping 512 square tiles of the 2 m orthophoto. Each
region's DAV2 prediction was fit by a separate affine transform to that same
region's DTM **only for an optimistic shape diagnostic**. The constant check
uses the mean DTM height from that region. Neither is deployable metric height.

| Statistic (region-level) | DAV2 Small | Constant |
| --- | ---: | ---: |
| Median oracle-aligned MAE | 184.032 m | 184.259 m |
| Median oracle-aligned RMSE | 222.175 m | 222.290 m |

DAV2 has lower MAE in 14/20 regions, with median paired relative reduction
0.179%. Its median oracle R² is 0.00248; the maximum is 0.0421. The model
therefore explains almost none of the within-region terrain variation in this
setup. Lower RMSE than a mean-height constant in all regions is guaranteed by
the per-region least-squares affine fit and is **not independent evidence**.

Per-region evidence is in `outputs/dav2_small/region_metrics.csv` and
`outputs/dav2_small/constant_metrics.csv`. Prediction GeoTIFFs are in the same
folder. Reproduce with `eval_dav2.py` followed by `eval_constant.py` using the
specified Python environments. The two-tile diagnostic is kept separately as
`pilot_metrics.csv` and must not enter summaries.

## Completed multi-method evaluation (2026-09-14)

All 20 regions now have region-wide DAV2 and Marigold V1.1 predictions on
the 2 m reference grid using 256 px information input per 512 px source tile.
Low-coverage edge windows were excluded (380,360 valid DTM pixels); the
remaining 758,340,549 pixels were scored under an identical mask.
The region-level median oracle MAE is 183.99 m (DAV2), 182.07 m (Marigold
V1.1), and 184.26 m (constant). V1.1 beats DAV2 in only 9/20 paired regions;
the paired relative improvement's region-bootstrap 95% interval spans zero.
See `outputs/dav2_small_input256/`, `outputs/marigold_v1_1_input256/`, and
`/mnt/d/nac/evaluation_20260914/summary_input256.json`.

Marigold V2 successfully ran on all **83** square inputs of a separate
**32 m/pixel** track covering all 20 regions. The official 2 m NAC ortho and
DTM were both averaged to this grid; inputs were reflect-padded to 256 square
and padding was cropped before scoring. DAV2 and Marigold V1.1 ran on the
same 83 images. Across 2,985,465 valid 32 m pixels, region-level median
oracle MAE is 159.54 m (DAV2), 184.40 m (V1.1), 171.53 m (V2), and 184.26 m
(constant). V2 beats DAV2 in 10/20 paired regions, with a 95% region-bootstrap
relative-improvement interval of -2.54% to +3.20%; no superiority claim is
supported. See `outputs/coarse32/` and
`/mnt/d/nac/evaluation_20260914/coarse32_summary.json`.

**All model scores are test-region oracle-affine shape diagnostics, not
deployable metric elevation.** The 32 m track does not test 2 m crater detail.
All 80 multi-column comparison PNGs and the 32 m reference/prediction GeoTIFFs
are under `/mnt/d/nac/evaluation_20260914/`.

## Supervised DMBNet-style adaptation, fold 0 pilot (2026-09-17)

Predeclared region-grouped 5-fold split in `outputs/supervised_splits.json`
(seed 20260913; 14 train / 2 val / 4 test regions per fold; every region
tested exactly once across folds). Fold 0 test regions: BHABHAE01, GODDARDA02,
GRUITHUISE4, RANGER9. Model: DMBDepth (`dmb_depth_model.py`) — frozen
precomputed DINOv3-S 4-stage features + trainable conv stem, per-stage MSA
fusion, progressive decoder, depth head + boundary-auxiliary BAM head; 47M
trainable parameters; 512 px tiles at 2 m. Trained on raw metric DTM targets
with masked Huber + gradient-L1 + boundary BCE, fp16 autocast, AdamW 3e-4,
early stop on the region-level val oracle-affine MAE (14 epochs, best epoch 5).

Region-level oracle-affine MAE on the four held-out test regions: 43.00
(BHABHAE01), 599.79 (GODDARDA02), 14.11 (GRUITHUISE4), 23.50 (RANGER9) m.
Against the paired constant baseline this is a mean +2.07 m improvement with a
region-bootstrap 95% CI of [0.01, 6.15] m — i.e. statistically marginal at
best, wins in 3/4 regions, median relative improvement +0.33%. DAV2 and
Marigold V1.1 sit within ~0.1% of the same numbers on these regions.

Diagnostic findings (fold 0):
- A per-tile-normalized training target teaches real within-tile shape
  (per-tile pred-DTM correlation 0.27-0.54 on test regions) but caps the
  region-level oracle MAE at ~constant: even a PERFECT per-tile-normalized
  predictor only improves region MAE by 1-6% (verified numerically), because
  per-tile mean offsets are unrecoverable from single 1 km tiles and dominate
  the region-level metric (GODDARDA02 tile-mean field MAD ~600 m).
- Training directly on raw metric targets (continuous frame) gives weak
  within-tile shape (corr 0.01-0.10): the Huber gradient is dominated by the
  unlearnable tile-mean offset error at every pixel.
- Conclusion: single-tile supervised adaptation reproduces the zero-shot
  result at region scale. The region-level metric requires cross-tile
  elevation continuity; candidate fixes are multi-tile context input and
  coarse topographic priors (LOLA/WAC), i.e. fusion with the思路-1 lunar
  foundation-model features.

Artifacts: `outputs/dmb_depth_fold0/` (checkpoint, train log, per-region
prediction GeoTIFFs, region_metrics.csv). Data: `/mnt/d/nac/supervised_v1/`
(materialized tiles + precomputed DINOv3 features; local working copy
`/home/zhy/lunar_data/supervised_v1`). Reproduce: `make_splits.py`,
`prepare_supervised_tiles.py`, `precompute_dinov3_features.py`,
`train_dmb_depth.py --fold 0`, `eval_dmb_depth.py --fold 0`.
