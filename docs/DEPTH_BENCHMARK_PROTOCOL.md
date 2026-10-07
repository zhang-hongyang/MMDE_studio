# KITTI / nuScenes five-method benchmark protocol

This release compares `unidepth_v2`, `moge3`, `ptc_dav2`, `mtd_dav2`, and
`manydepth2_vel` under one MMDE data and evaluation contract. Predictions are
stored as native-resolution `float32` metric-depth arrays. No median scaling
or sparse-GT fitting is applied to the reported raw metrics.

## Splits

| Dataset | Split | Frames | Evaluation |
|---|---:|---:|---|
| KITTI | `eigen_test` | 697 | public Eigen benchmark GT |
| KITTI | `official_test_anonymous` | 1000 | inference only; server GT is private |
| KITTI | `sequence_seed20261006_01..03` | 3 × 50 | held-out sparse LiDAR |
| nuScenes | `val_official` | 6019 | held-out projected LiDAR |
| nuScenes | `official_test` | 6008 | held-out projected LiDAR on v1.0-test images |
| nuScenes | `sequence_seed20261006_01..03` | 3 × 40 | held-out projected LiDAR |

The nuScenes official-test measurements are a local depth protocol on the
official v1.0-test sensor release, not a score returned by the nuScenes
challenge server. CAM_FRONT and LIDAR_TOP are public test inputs.

For every evaluable nuScenes frame, projected LiDAR returns are first rounded
to image pixels and de-duplicated by retaining the nearest return. A
deterministic, sample-token-specific permutation (global seed `20261006`)
assigns 20% of pixels to sparse controls and 80% to evaluation. Pixel overlap
between the two sets is prohibited and checked by the completion gate.

MMDE Studio exposes only two top-level test classifications. `test_single`
contains the official/evaluable single-frame subsets; `test_sequence`
contains the three deterministic continuous sequences. The physical split
name remains visible only as a second-level provenance selector so Eigen,
official-test, val, and the three sampled sequences are not conflated.

## Anonymous KITTI method modes

- UniDepthV2 and MoGe3 run their metric monocular paths.
- DAV2+MTD uses the official sparse depth input supplied with the 1000-frame
  depth-completion anonymous set.
- ManyDepth2-Vel uses its monocular branch because the selection contains no
  ordered temporal correspondence.
- DAV2+PTC is physically unobservable without temporal correspondence and a
  metric pose baseline. It is represented by explicit zero-depth invalid
  sentinels so requested-frame cardinality is preserved; it is not presented
  as a valid prediction.

## Metrics

Evaluation clips validity to `(0.1 m, 80 m)` and reports per-frame means and
pixel-pooled summaries. Primary raw metrics are AbsRel and AbsMean (metres),
with SqRel, RMSE, RMSE-log, and delta thresholds also retained. Pixel-pooled
AbsRel and AbsMean are additionally reported for depth bands `[0.1,10)`,
`[10,20)`, `[20,40)`, and `[40,80)` metres. Median-aligned metrics and
predicted/ground-truth scale dispersion are diagnostic fields only, not the
primary metric-depth claim.

## Scene visualization

Every valid `test_sequence` model result is fused with the recorded
`T_world_camera` poses into world-coordinate RGB point clouds. Fusion uses a
6-pixel image stride, 0.12 m voxel de-duplication, a 120k-point overview and
250k-point streamable detail chunks. The scene index retains camera position,
orientation, relative capture time, sequence bounds and exact point counts.
The one excluded combination is
`nuscenes/sequence_seed20261006_01/ptc_dav2`, whose 40 predictions are
explicit all-zero invalid sentinels; no empty or fabricated scene is emitted.

## Reproducibility gates

Completion requires exact frame and prediction cardinalities, a full finite
scan of every prediction array, zero control/evaluation pixel overlap, metric
JSON for every evaluable split/model pair, the MMDE Studio API contract test,
scene index/blob integrity for every valid sequence/model pair, and a
successful TypeScript/Vite production build.
