# The Midas Touch for Metric Depth (MTD)

[![arXiv](https://img.shields.io/badge/arXiv-2605.11578-b31b1b.svg)](https://arxiv.org/abs/2605.11578)
[![Project](https://img.shields.io/badge/GitHub-HenryMaxixi%2FMTD-blue)](https://github.com/HenryMaxixi/MTD)

Official implementation of **"The Midas Touch for Metric Depth"** (CVPR 2026 Highlight).

## Overall Framework



<p align="center">
  <img src="assets/overall_framework.png" alt="MTD Overall Framework" width="100%">
</p>

MTD takes relative depth, sparse 3D seeds, and a superpixel segment set as inputs and outputs reliable metric depth.

This release implements the pipeline with [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
as the depth foundation model, plus segment-wise affine fitting, graph propagation, recursive
bilateral filtering, and optional discontinuity-aware refinement (DADP).

---

## Status (v0.1)

- ✅ Single-image **demo** inference (`run_demo.py`)
- ⬜ Full segment / pixel fitting (under application)
- ⬜ Robust sparse-depth noise filtering

Due to an journal extension, the full codebase cannot be released at this time and will be released in the coming months — thank you for your understanding. You can contact mry5725@163.com.


---

## Repository layout

```
.
├── assets/demo/              # Demo RGB + sparse depth (user-provided)
├── data/kitti/               # KITTI DC val_selection_cropped (user-downloaded)
├── doc.md                    # Hyper-parameter guide for JSON configs
├── scripts/
│   └── download_checkpoints.sh
├── src/
│   ├── configs/
│   ├── core/                 # bilateral, dadp, superpixels, segment, …
│   ├── run_demo.py
│   └── run_kitti.py
└── third_party/
    └── Depth-Anything-V2/    # Git submodule
```

---

## Installation

### 1. Clone (with submodules)

```bash
git clone --recursive https://github.com/HenryMaxixi/MTD.git
cd MTD
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
pip install -r requirements.txt
```

### 2. Environment

Follow [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2):

```bash
pip install torch torchvision   # match your CUDA version
pip install -r requirements.txt
```

Optional backends (only if used in config):

```bash
pip install fast_slic              # segmentation.method = fast_slic
pip install opencv-contrib-python  # segmentation.method = opencv_lsc
```

### 3. Checkpoints

```bash
bash scripts/download_checkpoints.sh vitl
```

Weights are stored under `third_party/Depth-Anything-V2/checkpoints/`.

---

## Quick start — Demo

1. Place `image.jpg` and `sparse_depth.npy` under `assets/demo/example/` (or override via CLI).
2. Run:

```bash
cd src
python run_demo.py --config configs/demo.json
```

```bash
python run_demo.py --config configs/demo.json \
    --image  ../assets/demo/example/image.jpg \
    --sparse ../assets/demo/example/sparse_depth.npy
```

Output: `output/demo/`.

---

## KITTI Depth Completion evaluation

1. Download **val_selection_cropped** from the
   [KITTI Depth Completion benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).
2. Extract to `data/kitti/val_selection_cropped/` (see `data/kitti/README.md`).
3. Run:

```bash
cd src
python run_kitti.py --config configs/kitti_dc.json
```

---

## Configuration

All hyper-parameters live in `src/configs/*.json`.

- **Paths** (images, KITTI root, checkpoints): set in `dataset` / `model` / `output`.
- **Algorithm tuning** (segmentation, fitting, propagation, RBF, DADP): see **[doc.md](doc.md)**.

### Superpixel backends (`algorithm.segmentation.method`)

| `method` | Description |
|----------|-------------|
| `felzenszwalb` | Default; `scale`, `sigma`, `min_size` |
| `fast_slic` | [fast-slic](https://github.com/Algy/fast-slic); `num_components`, `compactness`, `use_avx2` |
| `opencv_lsc` | OpenCV LSC; `region_size`, `ratio`, `iterations` |

Example — switch to fast-slic in `demo.json`:

```json
"segmentation": {
    "method": "fast_slic",
    "fast_slic": {
        "num_components": 1600,
        "compactness": 10,
        "use_avx2": true
    }
}
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{midastouchmetricdepth,
      title={The Midas Touch for Metric Depth},
      author={Yu Ma and Zizhan Guo and Zuyi Xiong and Haoran Zhang and Yi Feng and Hongbo Zhao and Hanli Wang and Rui Fan},
      year={2026},
      url={https://arxiv.org/abs/2605.11578},
}
```

---

## Acknowledgements

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — monocular depth backbone
- [fast-slic](https://github.com/Algy/fast-slic) — optional superpixel segmentation
- [KITTI Depth Completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) — benchmark data

---

## License

MTD source code is released under the MIT License (see `LICENSE`).

Depth Anything V2 and its pretrained weights are subject to their own licenses; weights are not redistributed in this repository.
