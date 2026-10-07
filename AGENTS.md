# MMDE-Studio 项目指南

MMDE（单目深度估计研究平台）的 B/S Web 端。替代旧查看器
`/home/MMDE/mmde/scripts/pc_viewer_server.py`（旧文件保留不动）。

## 结构

- `algorithms/` — 单目深度算法仓库的家（每个算法一个子目录，完整第三方/自研
  repo，平级共享）。已入驻：`marigold-v2`。算法专属 Python 环境放
  `algorithms/.venv_<name>`（机器本地，不进版本库，迁移后按各 repo 的
  setup 重建 + 更新 `methods/interpreters.yaml`）。
- `methods/` — 任务中心适配层：`run_<method>.py` 是轻量包装脚本（只需后端
  依赖 yaml+numpy），负责 staging 数据集帧 → 调用 `algorithms/<name>` 的推理
  入口（子进程，解释器来自 `interpreters.yaml` 或 `<METHOD>_PYTHON` 环境变量）
  → 把预测写成 MMDE preds 树 `{pred_root}/{split}/{model_name}/{idx:06d}.npy`。
  `interpreters.yaml` 是与 `datasets.yaml` 同级的机器本地配置（唯一允许出现
  机器相关解释器路径的地方）。任务中心按 `run_*.py` glob 自动发现方法。
- `backend/` — FastAPI（Python；本机用 conda env `lunarecon`，解释器
  `/home/zhy/miniconda3/envs/lunarecon/bin/python`，`MMDE_PYTHON` 可覆盖；
  原 `dggt` 环境只在 /root 的那台机器上存在）
  - `config/datasets.yaml` — 数据集注册表（platform: vehicle/uav/satellite）；splits/cameras/控制点源/模型/场景全部自动发现
  - `app/registry.py` / `app/data_access.py` / `app/routers/` / `app/tasks/`
  - `tests/` — pytest 契约测试（对真实只读数据）；`docs/API.md` — API 与二进制格式
- `frontend/` — React 18 + TS strict + Vite + Tailwind v4 + react-three-fiber + zustand
  - `src/styles/tokens.css` — 三层 design token（暗色默认，规范见 `frontend/docs/design/MASTER.md`）
  - `src/components/ui/` — 手写 shadcn 风格组件；`src/features/{explore,scene,controls,metrics,tasks}/` — 页面
- `scripts/dev.sh` — 开发（后端 :8010 --reload + 前端 vite :5173，/api 代理）；`scripts/start.sh` — 生产（单端口 8010）

## 命令

```bash
./scripts/dev.sh                                                     # 开发
./scripts/start.sh                                                   # 生产（PORT=xxx 可改）
${MMDE_PYTHON:-/home/zhy/miniconda3/envs/lunarecon/bin/python} -m pytest backend/tests -q   # 后端测试
cd frontend && npm install && npm run build                          # 前端构建
```

注意：**本机 8000 被其他服务占用**（返回 401），本项目固定 8010；旧 pc_viewer 仍在 8765。
后端磁盘缓存在 `backend/.cache/`（可整目录删除）；任务库 `backend/.cache/tasks.db`。

## 约定

- **禁止**在前后端代码中硬编码数据集名/绝对数据路径（唯一例外：`backend/config/datasets.yaml` 与
  datasets.yaml 里的路径）；新增数据集 = 数据按 frames.jsonl 约定入目录 + yaml 加一段。
- **禁止**改动 `/home/MMDE`、`/home/data2` 下的任何文件（后端只读 + subprocess 执行其脚本）。
- 组件内禁止裸 hex，一律 design token；新页面先读 `frontend/docs/design/MASTER.md`。
- 二进制 API 变更必须递增 `X-Format-Version` 并更新 `backend/docs/API.md`。
- 前端状态用 zustand 按域拆分；URL 同步页面选择状态（可分享/刷新恢复）。
- 后端测试必须保持全绿才算完成；前端 `tsc --noEmit` + `npm run build` 零错误才算完成。
- 任务执行一律 subprocess 参数列表形式（禁 shell=True），conda python 路径可配。
- 外部脚本位置与环境变量（迁移部署时只改环境、不改代码，详见
  `backend/app/tasks/catalog.py` 头部注释）：
  - `MMDE_PYTHON` — 任务子进程解释器（dev.sh/start.sh 同样读取并 export）
  - `MMDE_METHODS_DIR` / `MMDE_EVAL_SCRIPT` / `MMDE_FUSE_SCRIPT` — 推理/评测/融合脚本路径
  - `MMDE_STUDIO_DATASETS` / `MMDE_STUDIO_MODELS` — 数据集/模型注册表 yaml

## 接入一个新深度算法（重复此流程）

1. 算法仓库整体放入 `algorithms/<name>/`（其内路径全部相对自身，符合
   /home/research/CLAUDE.md 规范）。
2. 按该仓库的 setup 说明建环境 `algorithms/.venv_<name>`（或复用现有 conda
   env），确认其推理入口 CLI（如 `scripts/infer.py`）。
3. 写 `methods/run_<name>.py` 包装：argparse **必须声明** `--dataset`、`--split`
   （catalog 按此过滤传参；可选 `--max-frames`/`--model-name`/`--device`），
   从 `MMDE_STUDIO_DATASETS`（默认 `backend/config/datasets.yaml`）读
   test_root/pred_root，staging `frames.jsonl` 的图像，子进程跑算法解释器，
   预测写回 `{pred_root}/{split}/{model_name}/{idx:06d}.npy`（形状须与数据集
   图像一致，float32；相对深度即可，服务时逐帧仿射对齐到稀疏 GT）。
4. 在 `methods/interpreters.yaml` 登记 `<name>: <解释器路径>`。
5. `backend/config/models.yaml` 给 `model_name` 加分类规则（前端下拉分组）。
6. 契约测试自动覆盖：`test_method_wrappers_declare_contract_flags` 要求每个
   `run_*.py` 声明 `--dataset`/`--split`；`pytest backend/tests -q` 须全绿。
7. 网页 Tasks → Depth inference 选新方法冒烟（`max_frames=2`），确认
   `/api/points/.../<model_name>` 返回 200 且 depth_aligned 落在 GT 量级。
