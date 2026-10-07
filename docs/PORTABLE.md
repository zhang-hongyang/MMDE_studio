# MMDE-Studio 导出部署指南（PORTABLE）

本文档面向**新环境部署**：把 MMDE-Studio 代码包解压到一台新机器上，配置数据路径后快速跑通。
代码包只含 Web 端本体（`backend/` + `frontend/` + `scripts/` + 文档），**不含数据集与模型权重**。

## 1. 包里有什么

```
studio/
├── backend/               FastAPI 后端
│   ├── config/datasets.yaml    数据集注册表（部署时必改，见 §3）
│   ├── config/models.yaml      模型下拉分组规则（通配规则，按需调整）
│   ├── app/                    registry / data_access / routers / tasks
│   ├── tests/                  pytest 契约测试（46 项，含数据可用性跳过）
│   ├── requirements.txt        后端最小依赖
│   └── docs/API.md             REST 与二进制格式契约
├── frontend/              React SPA（已含 dist/ 预构建产物，可免 node 直接跑）
├── scripts/dev.sh         开发模式（后端 :8010 --reload + 前端 vite :5173）
├── scripts/start.sh       生产模式（npm ci + build，单端口 8010 托管全部）
├── README.md / AGENTS.md  项目说明与开发约定
└── frontend/docs/design/MASTER.md   UI 设计系统规范
```

## 2. 环境要求

| 依赖 | 版本 | 用途 | 必需性 |
|---|---|---|---|
| Python | ≥ 3.10 | 后端 | 必需 |
| pip 包 | 见 `backend/requirements.txt` | fastapi/uvicorn/numpy/opencv-headless/scipy/pyyaml | 必需 |
| Node.js + npm | ≥ 20 | 前端构建 | 可选（有 dist 可跳过；改前端才需要） |
| GPU + torch 等 | CUDA 机器、与方法匹配的环境 | 网页发起推理/评测任务 | 仅 Tasks 功能需要 |

最小安装：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
```

## 3. 快速跑通（三选一）

### A. 只看界面结构（零数据）
直接启动，注册表里的数据集路径不存在会被自动跳过，UI 正常打开、显示空态：

```bash
./scripts/start.sh        # 或 PORT=8080 ./scripts/start.sh
# 打开 http://localhost:8010
```

### B. 接入已有 MMDE 数据（推荐）
数据沿用 MMDE 统一约定（frames.jsonl 索引，见 README）。复制 `backend/config/datasets.yaml`，
把 `test_root / pred_root / metrics_root / scene_root` 改成你的实际路径，然后：

```bash
export MMDE_STUDIO_DATASETS=/path/to/your/datasets.yaml   # 不拷原文件，用环境变量指向
export MMDE_PYTHON=/path/to/your/python                   # 任务子进程解释器（见 §5）
./scripts/start.sh
```

数据集/split/相机/控制点源/模型/场景全部**自动发现**：数据入目录即出现，无需改代码。
新增无人机/卫星数据：数据按同一 frames.jsonl 约定组织 + yaml 里加一段（`platform: uav/satellite`）。

### C. 前端也要改（开发模式）
```bash
cd frontend && npm install && cd ..
./scripts/dev.sh
```

## 4. 数据布局约定（接入自己的数据时唯一要懂的东西）

```
{test_root}/{dataset}/{split}/
├── frames.jsonl          # 每帧一行 JSON（唯一索引）：
│                         #   seq_id, frame_id, timestamp, camera?,
│                         #   image_path,           # RGB 原图（绝对路径或迁移后存在路径）
│                         #   K(3x3), T_world_camera(4x4, camera→world),
│                         #   speed_mps, depth_gt_path?, calib_seq_path?, calib_mono_feye_path?
├── depth_gt/{seq}/       # 稀疏 LiDAR GT .npz（uv N×2 f32, depth N f32）— 可选
└── sparse_controls/{src}/{idx:06d}.npz  # 控制点（uv/depth/weight）— 可选
{pred_root}/{dataset}/{split}/{model}/{idx:06d}.npy   # 深度预测（视差模型见 §6）
{metrics_root}/{dataset}/{split}/{model}[.standard].json  # 指标（eval 产物）
{scene_root}/{dataset}/{split}/{model}/               # 融合场景（fuse_scene 产物）
```

## 5. 网页任务中心（Tasks）的依赖 —— 三方对比方法全清单

Tasks 页通过 subprocess 调用 **MMDE 研究仓库**的脚本执行推理/评测/融合，不内嵌任何模型代码。
迁移时脚本位置与环境变量（`backend/app/tasks/catalog.py` 头部注释为准）：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `MMDE_PYTHON` | `/root/miniconda3/envs/dggt/bin/python` | 任务子进程解释器（需装有 torch 及方法依赖） |
| `MMDE_METHODS_DIR` | `/home/MMDE/mmde/methods` | 推理脚本目录（glob `run_*.py` 自动发现方法） |
| `MMDE_EVAL_SCRIPT` | `/home/MMDE/mmde/eval/eval_mmde.py` | 统一评测脚本 |
| `MMDE_FUSE_SCRIPT` | `/home/MMDE/mmde/scripts/fuse_scene.py` | 场景融合脚本 |
| `MMDE_STUDIO_DATASETS` / `MMDE_STUDIO_MODELS` | `backend/config/*.yaml` | 注册表覆盖 |

**路径不存在时对应任务类型自动从任务向导中消失**（catalog 容错），浏览可视化功能不受影响。
推理脚本自身再用 `MMDE_TEST_ROOT / MMDE_RESULT_ROOT / MMDE_SCENE_ROOT / MMDE_DGGT_ROOT`
等环境变量解析数据与三方仓库位置（见 MMDE 仓库 `mmde/paths.py`）——**迁移只改环境，不改代码**。

### 5.1 当前可执行推理方法（`methods/run_*.py`，7 个）

Web 向导只暴露 method/dataset/split + 少量通用 flag；构建命令时会扫描所选脚本源码、
只发射该脚本真实声明的 flag，所以下表"关键参数"以外的调整请改脚本默认值。

| 方法脚本 | 下拉分组 | 方法说明 | 关键 CLI 参数 | 依赖的三方仓库（3rdparty 软链） |
|---|---|---|---|---|
| `run_marigold_v2.py` | Marigold 零样本 | 已验证的 Marigold V2 单目相对深度 | `--dataset --split --max-frames --model-name --device` | `marigold-v2` |
| `run_moge3.py` | 度量模型 | MoGe-3 ViT-L；使用相机内参计算 FOV，输出原分辨率米制深度 | 同上 | `moge` |
| `run_mapanything.py` | 前馈式 | MapAnything 多帧前馈式米制三维重建，并抽取每视图 Z-depth | 同上 | `map-anything` |
| `run_lingbot_map.py` | 前馈式 | LingBot-Map 按 `seq_id` 流式前馈重建；MMDE 采用 SDPA 路径并保存每帧 depth | 同上 | `lingbot-map` |
| `run_ptc_dav2.py` | PTC | PTC-Depth + DAV2；从相邻 `T_world_camera` 平移计算 baseline；首帧是明确的 warm-up，不伪造零深度结果 | 同上 | `ptc-depth` + MTD 内 DAV2 |
| `run_mtd_dav2.py` | MTD | MTD + DAV2；将 `depth_gt_path` 的稀疏 LiDAR 投影作为 metric seeds | 同上 | `mtd` + DAV2 |
| `run_mtd_moge3.py` | MTD | MMDE 组合链：复用 MoGe-3 米制预测，转 inverse-depth proxy 后交给 MTD 标定 | 同上 | `moge` + `mtd` |

三方源码提交固定在 `algorithms/SOURCES.lock.yaml`。8 卡机上各依赖族使用独立环境
`mmde-moge3 / mmde-mapanything / mmde-lingbot / mmde-temporal`，实际解释器只在
`methods/interpreters.yaml` 这一部署配置中声明；模型权重统一放在
`ResearchHub-scratch/models/MMDE_studio/<method>/`。

PTC/MTD 共享的 `mmde-temporal` 环境固定为 CPython 3.10、OpenSSL 3.x、
OpenCV 4.x、Eigen 3.x，并由轻量 launcher 仅对 temporal 子进程注入该环境的
`lib/` 到 `LD_LIBRARY_PATH`，避免修改服务器系统 C++ 运行库。`fast_slic` 和
`opencv-contrib-python` 是可选 superpixel 后端；默认、可复现的 MTD 路径使用
scikit-image Felzenszwalb，因此不下载未启用的 contrib 包。

> 边界说明：MTD 上游 v0.1 目前明确标注“完整 segment/pixel fitting 尚未公开”。
> `mtd_dav2` 严格使用其当前公开实现；`mtd_moge3` 是本项目的可审计组合实验，
> 不是 MTD 论文作者发布的原生 MoGe-3 backbone。使用同一稀疏 LiDAR 同时作为
> 输入 seed 和评测 GT 会产生协议泄漏，正式指标必须采用独立 hold-out GT。

### 5.2 评测与场景融合

| 任务类型 | 脚本 | 说明 | 参数 |
|---|---|---|---|
| eval | `mmde/eval/eval_mmde.py` | 统一评测：原始/尺度/仿射/视差/序列仿射 5 种协议，输出 `metrics/*.json`（Metrics 页数据源） | `--dataset(kitti/nuscenes/carizon) --split --models… --protocol(mmde/standard)` |
| fuse_scene | `mmde/scripts/fuse_scene.py` | 多帧深度融合成场景点云块（Scene 页数据源） | `--dataset --split --model --voxel --stride --min-depth --max-depth --max-frames --force` |

### 5.3 任务系统行为

- 队列进程内调度、SQLite 持久化（`backend/.cache/tasks.db`），并发上限 1（`tasks_concurrency` 可配，防 GPU 争抢）
- 日志逐行 WebSocket 推送（`/ws/tasks/{id}`，前端断线自动降级轮询）；cancel 杀整个进程组
- 安全：参数白名单 + NAME_RE 校验 + argv 列表执行（无 shell 注入面）

## 6. 模型分组与视差约定

- 下拉分组由 `backend/config/models.yaml` 通配规则驱动（顺序首个命中）：MTD / PTC / TTA / 前馈式 / 度量模型 / 其他；
  `excluded`（`dav2_rel* zipdepth* gemdepth*`）在 Web 端隐藏（相对深度结果不展示）。
- **视差约定**：`dav2_rel / zipdepth / gemdepth` 三个模型的原始 .npy 存的是逆深度，消费端统一 `1/max(x,1e-6)` 转深度；
  `*_calib_*` 输出已是米制深度，**不可再反转**（后端 `DISP_MODELS` 集合，与 `eval_mmde.py IS_DISP` 保持一致）。
- 修改分组/排除：只改 `models.yaml`，前端运行时自动生效，无需构建。

## 7. 验证与故障排查

```bash
pip install -r backend/requirements.txt pytest httpx
python -m pytest backend/tests -q        # 43 passed + 数据缺失项 skip 即正常
```

| 现象 | 原因与处理 |
|---|---|
| Overview 空、下拉无数据集 | datasets.yaml 路径不对；`export MMDE_STUDIO_DATASETS` 指向正确 yaml |
| 数据集在但点云/图像 404 | frames.jsonl 里 image_path 是录制机绝对路径，需在新环境可访问（或重建 frames.jsonl） |
| Tasks 向导没有推理/评测类型 | `MMDE_METHODS_DIR / MMDE_EVAL_SCRIPT / MMDE_FUSE_SCRIPT` 路径不存在，检查环境变量 |
| 任务秒失败 exit≠0 | 看任务日志；多为 `MMDE_PYTHON` 环境缺 torch/对应三方仓库或 GPU |
| 8010 被占 | `PORT=xxxx ./scripts/start.sh` |
| 前端要改版 | `cd frontend && npm install && npm run build`（或 `./scripts/dev.sh`） |
| 浏览器打不开 WebGL | 需要支持 WebGL2 的浏览器（Chrome/Edge/Firefox 均可） |

## 8. 版本与再打包

```bash
# 在 studio/ 父目录执行（排除依赖与缓存，含前端 dist）
tar --exclude='studio/frontend/node_modules' --exclude='studio/backend/.cache' \
    --exclude='*/__pycache__' --exclude='*.pyc' --exclude='studio/.pytest_cache' \
    -czf mmde-studio-$(date +%Y%m%d).tar.gz studio/
```
