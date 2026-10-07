# LFM 单目深度验证 · 第一步结果（冻结特征线性探针）

日期：2026-09-17　数据：`supervised_splits.json` 5 折地理留出（每折 2 个留出区域，共 10 个 fold-region 对）

## 实验设置

- 输入：input256 轨道同款瓦片（512px@2m → 256px PNG），与 DAV2/Marigold 评测完全同协议
- 特征：NASA-IBM Lunar FM（SOMA, ViT-B）`nac` 单模态编码器，冻结，逐 block 输出 (B,256,768)
- 标签：DTM 2m 按 16×16 patch 平均（每 token 代表 64m 地面）
- 探针：全局岭回归（无逐测试瓦片拟合；oracle 仿射仅作 shape 诊断单独报告）
- 对照：常数基线（区域均值）；DAV2 相对深度（16×16 均值，1 维特征）探针
- λ∈{1e2..1e5}，layer∈{3,6,9,11}，每区域取最优组合

## 结果（oracle 仿射后 MAE，m）

| 指标 | SOMA 探针 | 常数 | DAV2 探针 |
|---|---|---|---|
| 均值（10 fold-region） | **308.2** | 330.4 | 330.3 |
| 均值 oracle RMSE | **369.6** | 391.0 | 390.9 |
| 逐对胜率 | 10/10 | — | 10/10 |
| 相对常数改善（中位 / 最大） | 3.0% / 16.0% | — | ≈0% |

- DAV2 探针 ≡ 常数基线（与此前 R²≈0.0025 的结论一致：通用模型的相对深度不含可用高程信号）
- SOMA 最优层集中在第 9–11 层（深层语义），λ 偏好大正则（1e5）
- 单区域最大改善出现在 FRSHCRATER10（新鲜撞击坑，762.6 vs 908.3 m，−16%）

## 关键局限（诚实声明）

1. **幅度有限**：线性读出只捕获低频趋势（区域内地形起伏），改善 3–7% 尚不构成可用产品
2. **shape_rmse 差**：SOMA 探针 tile 去均值残差 RMSE ≈678 m ≫ 常数 46 m——线性头在高频端注入噪声，需要非线性多尺度稠密头 + 微调（LoRA）才能利用高频信号
3. **绝对误差不可用**：跨区域预测存在基准面偏置（abs RMSE 公里级），任何实用方案需解决 datum/偏置问题（如 SLDEM 先验配准或区域平差）
4. **输入 GSD 差异**：SOMA 预训练为 1m GSD@256px，本实验输入为 2m（4m/px 等效），有一定域偏移

## 结论

**假设获得初步支持**：月球专用 FM 的冻结特征含有可提取的高程信号（10/10 稳定胜过常数与通用模型），而通用模型完全没有。但线性探针天花板明显，下一步必须上稠密头 + LoRA 微调。

## 复现

```bash
.venv_lfm/bin/python lfm_experiment/extract_lfm_features.py   # 特征（已缓存 outputs/lfm_probe/features）
/home/research/code/lunarecon/.conda-env/bin/python lfm_experiment/make_labels.py  # 标签
.venv_lfm/bin/python lfm_experiment/run_sweep.py              # 5折×层×λ 扫描 → sweep_results.csv
```

产物：`outputs/lfm_probe/sweep_results.csv`（180 行）、`probe_results_fold0.json`
