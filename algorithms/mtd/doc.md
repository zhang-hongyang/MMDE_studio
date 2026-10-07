# Hyper-parameter guide (`src/configs/*.json`)

This document explains **algorithm hyper-parameters** in the JSON configs (not file paths).
See [README.md](README.md) for installation and dataset layout.

Configs: `src/configs/demo.json` (single image) and `src/configs/kitti_dc.json` (KITTI evaluation).

---

## `algorithm.min_depth` / `algorithm.max_depth`

Valid depth range in **metres**. Predictions and metrics are clipped to `[min_depth, max_depth]`.

- **Demo** (indoor / short range): e.g. `0.1` – `15.0`
- **KITTI** (outdoor driving): e.g. `2.0` – `78.0`

---

## `algorithm.segmentation`

Chooses how the image is partitioned into superpixels before per-region affine fitting.

> **Runtime note:** Superpixel segmentation runs on the CPU at the full input resolution and is often the main bottleneck (especially `felzenszwalb` on large images). To speed up inference, downsample the RGB image and sparse depth to a smaller size before running (keep both aligned), or switch to `fast_slic`. 

### `method`

| Value | Library | Notes |
|-------|---------|--------|
| `felzenszwalb` | scikit-image | Default; good edge alignment; tune `scale` / `sigma` / `min_size` |
| `fast_slic` | [fast-slic](https://github.com/Algy/fast-slic) | Very fast CPU SLIC; tune `num_components` / `compactness` |
| `opencv_lsc` | OpenCV `ximgproc` | Needs `opencv-contrib-python`; `region_size` / `ratio` |

Example — Felzenszwalb (demo-style, fewer large segments):

```json
"segmentation": {
    "method": "felzenszwalb",
    "scale": 1000,
    "sigma": 0.5,
    "min_size": 1000
}
```

Example — fast-slic (as in internal experiments):

```json
"segmentation": {
    "method": "fast_slic",
    "fast_slic": {
        "num_components": 1600,
        "compactness": 10,
        "use_avx2": true,
        "min_size_factor": 0
    }
}
```

Example — OpenCV LSC:

```json
"segmentation": {
    "method": "opencv_lsc",
    "opencv_lsc": {
        "region_size": 20,
        "ratio": 0.075,
        "iterations": 1
    }
}
```

#### Felzenszwalb-only keys

| Key | Effect |
|-----|--------|
| `scale` | Larger → fewer, coarser superpixels |
| `sigma` | Gaussian smoothing before graph segmentation |
| `min_size` | Minimum segment area (pixels); small regions are merged |

#### fast-slic-only keys (`fast_slic` sub-object or top-level)

| Key | Effect |
|-----|--------|
| `num_components` | Target number of superpixels |
| `compactness` | Colour vs spatial balance (higher → more compact blobs) |
| `use_avx2` | Use `SlicAvx2` when CPU supports AVX2 (faster) |
| `min_size_factor` | Post-filter tiny regions; `0` skips denoising (faster) |

#### OpenCV LSC-only keys (`opencv_lsc` sub-object)

| Key | Effect |
|-----|--------|
| `region_size` | Initial superpixel size |
| `ratio` | Colour vs spatial weight |
| `iterations` | SLIC iterations (often `1`) |

---

## `algorithm.segment_fitting.min_count`

Minimum number of **valid sparse LiDAR points** inside a superpixel to fit the affine
disparity model `d_sparse ≈ a · d_rel + b`.

- Too **low** → noisy least-squares fits on tiny regions.
- Too **high** → many segments skipped; more reliance on propagation.

Typical values: `5` (demo), `3` (KITTI sparse velodyne).

---

## `algorithm.segment_backend`

| Value | Behaviour |
|-------|-----------|
| `torch` | Parallel GPU OLS per segment (faster) |
| `numpy` | sklearn `LinearRegression` per segment (matches original benchmark scripts; often slightly better metrics on KITTI) |

---

## `algorithm.use_segment_propagation`

When `true`, segments **without** enough sparse points receive scale or affine parameters
propagated along a **k-NN segment graph** built from region adjacency / colour similarity.

When `false`, a single **global median ratio** between sparse and relative disparity is used
(cheaper, less accurate on distant objects).

---

## `algorithm.use_params_propagation`

Only used when `use_segment_propagation` is `true`.

| Value | Propagation target | Pipeline |
|-------|-------------------|----------|
| `false` | Median scale ratio **T** per segment | Default; depth-level bilateral filter after merge |
| `true` | Affine **a**, **b** per segment | RBF smooths a/b maps; skips depth-level RBF |

Use `false` for the original T-propagation path; try `true` when boundaries need per-segment affine variation.

---

## `algorithm.propagation.k_neighbors`

Number of nearest segment neighbours in the propagation graph (typical: `4`).

---

## `algorithm.bilateral_filter`

Recursive bilateral filter (RBF) guided by RGB.

| Key | Effect |
|-----|--------|
| `iterations` | More passes → smoother depth / parameter maps |
| `spatial_sigma` | Spatial support of the kernel |
| `color_sigma` | Edge sensitivity (smaller → sharper preservation of image edges) |
| `kernel_size` | Patch size (odd integer) |
| `use_anchor` | If `true`, LiDAR-fitted pixels stay fixed while neighbouring pixels are filled |

With `use_anchor: true`, stronger smoothing
(more `iterations`, larger `spatial_sigma` / `kernel_size`) propagates depth from sparse anchors
over a wider neighbourhood. That reduces holes and noise but can **soften depth boundaries** near
object edges. Weaker smoothing keeps edges sharper but leaves **more high-frequency noise and speckle**
in filled regions. Adjust `iterations`, `spatial_sigma`, and `color_sigma` together rather than
`use_anchor` alone.


---

## `algorithm.use_dadp`

Enable **Discontinuity-Aware Dynamic Programming** refinement after merging sparse seeds.

- **Demo**: usually `false` (faster, sufficient indoors).
- **KITTI**: often `true` for sharper depth discontinuities.

---

## `algorithm.dadp_variant` / `algorithm.dadp`

| `dadp_variant` | Description |
|----------------|-------------|
| `torch_dadp` | GPU Taylor-style candidate interpolation (default) |
| `numpy_dadp` | CPU single-pass (slower) |

| `dadp` key | Effect |
|------------|--------|
| `iterations` | iterations; `1` is usually enough for `torch_dadp` |
| `seed_energy` | Energy bias keeping sparse measurements fixed (e.g. `-10.0`) |
