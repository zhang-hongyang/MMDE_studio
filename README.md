# MMDE-Studio

单目深度估计研究平台 MMDE 的 Web 端（B/S 架构），包含前后端、算法适配层、
评测与场景融合脚本。替代旧的单文件查看器
（`/home/MMDE/mmde/scripts/pc_viewer_server.py` + `pc_viewer.html`，保留不删，功能迁移完成后停用）。

设计目标：

- **平台可扩展**：数据集注册表驱动，新增车载/无人机/卫星影像 = 数据按约定入目录 + `backend/config/datasets.yaml` 加一段配置，前端零改动。
- **功能分层**：路由化信息架构——Overview / Explorer（帧 A/B 对比）/ Scene（融合场景点云 + 车载回放）/ Controls（稀疏控制点诊断）/ Metrics（指标看板）/ Tasks（任务中心：网页发起推理/评测/场景融合）。
- **设计系统**：遵循 [ui-ux-pro-max-skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill) 规范，Dark-first 双主题，三层 design token，规范存档于 `frontend/docs/design/MASTER.md`。

## 架构

```
frontend/  React 18 + TS + Vite + Tailwind v4 + react-three-fiber (npm)
backend/   FastAPI + uvicorn (/root/miniconda3/envs/dggt 的 Python)
数据        只读复用 MMDE 约定，不搬迁：
  /home/data2/mmde_test/{ds}/{split}/        frames.jsonl / depth_gt / sparse_controls
  /home/data2/mmde_result/preds|metrics/     预测 .npy / 指标 .json
  /home/MMDE/data_scene/{ds}/{split}/{model}/ 融合场景 index.json + chunks
```

## 快速开始

```bash
# 安装前端依赖，并按本机环境配置算法解释器
cd frontend && npm install && cd ..
cp methods/interpreters.example.yaml methods/interpreters.yaml

# 开发（后端 uvicorn --reload :8010 + 前端 vite :5173，/api 自动代理）
./scripts/dev.sh

# 生产（build 前端，FastAPI 单端口托管全部）
./scripts/start.sh          # PORT 环境变量可改端口，默认 8010
```

注意：本机 8000 端口被其他服务占用（返回 401），MMDE-Studio 固定使用 8010。
远程访问：`ssh -L 8010:localhost:8010 user@host`。

## 新增单目深度算法

算法仓库放 `algorithms/<name>/`（如 `algorithms/marigold-v2`），专属环境放
`algorithms/.venv_<name>`；`methods/run_<name>.py` 是轻量包装（staging 帧 →
子进程调算法推理 → 写 preds 树），`methods/interpreters.yaml` 登记各方法的
解释器。任务中心按 `run_*.py` 自动发现方法，网页 Tasks 页即可发起推理。
详细步骤见 `AGENTS.md` 的「接入一个新深度算法」。

仓库只保存代码与可复现配置模板。数据集、模型权重、预测结果、融合场景、
Python/Node 环境和机器专属的 `methods/interpreters.yaml` 均不进入 Git；运行时通过
`MMDE_STUDIO_DATASETS`、`MMDE_STUDIO_MODELS`、`MMDE_PYTHON` 等环境变量接入。

## 新增数据集 / 新平台（车载 → 无人机 → 卫星）

1. 数据按统一约定组织（与现有数据集同构）：
   ```
   {test_root}/{dataset}/{split}/frames.jsonl   # 每行: seq_id, frame_id, timestamp, camera?,
                                                #       image_path, K, T_world_camera, speed_mps,
                                                #       depth_gt_path, calib_seq_path, ...
   ```
2. 在 `backend/config/datasets.yaml` 增加一段（`platform: vehicle | uav | satellite`，
   splits/cameras/控制点源/模型全部自动发现，无需逐一声明）。
3. 刷新网页即出现在所有下拉框与 Overview 卡片中。前端如需平台专属面板（如卫星的地理坐标显示），
   按 `platform` 字段懒加载对应组件。

## 测试与文档

```bash
# 测试（后端契约测试，对真实只读数据；Python 可用 MMDE_PYTHON 覆盖）
${MMDE_PYTHON:-/home/zhy/miniconda3/envs/lunarecon/bin/python} -m pytest backend/tests -q
cd frontend && npm run build                                       # 前端构建
```

本机部署说明（WSL，`/home/zhy`）：后端解释器默认 `lunarecon` conda 环境
（`scripts/dev.sh` / `start.sh` 读取 `MMDE_PYTHON` 覆盖）。示例数据集为
`lunar_nac`（20 个 LROC NAC 区域 + DAV2/Marigold 预测，platform: satellite），
由 `scripts/prepare_lunar_dataset.py` 从 `/mnt/d/nac/official_rdr/expanded` 与
`lunar_nac_eval/outputs` 生成到 `data/`；原四个车载数据集在 `/home/data2`
不可用时自动跳过。

- API 契约与二进制格式：`backend/docs/API.md`
- 设计系统规范：`frontend/docs/design/MASTER.md`（M4 补充）
- KITTI / nuScenes 五方法评测协议：`docs/DEPTH_BENCHMARK_PROTOCOL.md`

## 与旧版对照

旧 pc_viewer 功能在新版的入口：

| 旧版功能 | 新版入口 |
|---|---|
| 模型 A/B 对比、帧间差分、叠加、ROI、着色 | Explorer |
| 场景点云、轨迹、fly-through 回放 | Scene |
| 控制点叠加与诊断 | Controls |
| （无）指标查看 | Metrics |
| （无）网页发起推理/评测 | Tasks |
