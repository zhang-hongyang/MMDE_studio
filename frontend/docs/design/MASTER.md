# MMDE-Studio Design System — MASTER

唯一事实源。按 [ui-ux-pro-max-skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill) 工作流生成
（查询: "scientific data visualization research dashboard"，density 8，motion Subtle 档）。
后续新增页面/组件必须先读本文件；页面级覆盖写 `pages/<page>.md` 覆盖对应章节，不得改本文件。

- **Pattern**: Enterprise Gateway（顶部工具栏 + 左侧导航 + 内容区）
- **Style**: Data-Dense Dashboard（主） + Dark Mode OLED（副）
- **产品类型**: Analytics Dashboard / 科学数据可视化工具
- **原则**: Clarity > aesthetics. Color-coded data priority. 不做 ornate 装饰、不过度过滤功能。

## Colors（语义 token，定义于 `src/styles/tokens.css`）

| Token | 暗色 | 亮色 | 用途 |
|---|---|---|---|
| --bg | `#0B1220` | `#F8FAFC` | 页面背景 |
| --card | `#111A2C` | `#FFFFFF` | 卡片/面板 |
| --card-2 | `#0E1626` | `#F1F5F9` | hover/次级面 |
| --border | `#1F2A44` | `#E2E8F0` | 1px 边框 |
| --fg | `#E6EDF7` | `#0F172A` | 正文 |
| --fg-muted | `#8B98AD` | `#64748B` | 次要文字 |
| --primary | `#3B82F6` | `#1D4ED8` | 主操作/数据蓝 |
| --on-primary | `#FFFFFF` | `#FFFFFF` | primary 上文字 |
| --accent | `#D97706` | `#B45309` | 高亮/选中/A 序列 |
| --success / --danger / --warning | `#22C55E` / `#EF4444` / `#F59E0B`（两主题不变） | 同左 | 状态语义色 |

规则：组件内**禁止裸 hex**，一律 `var(--token)`；语义状态色必须配图标或文字，禁止纯颜色编码；
暗色不是亮色反色，是两组独立映射；3D 视口内数据色 A=accent 橙 / B=primary 蓝（Okabe-Ito 兼容）。

## Typography

- UI/正文：**Fira Sans**（@fontsource/fira-sans）；字号阶梯 12 / 13 / 14 / 16 / 18 / 24；正文行高 1.5。
- 数据/等宽：**Fira Code**（@fontsource/fira-code）——帧号、指标数值、日志、状态栏；数字一律
  `font-variant-numeric: tabular-nums` 防抖动。
- 3D 视口 HUD：Fira Code 12px。

## Density（Data-Dense Dashboard tokens）

sidebar 宽 **240px**；header 高 **56px**；卡片 padding **12px**；内容区间距 **8px**；
表格行高 **36px**（sticky 表头、可排序、数字 tabular-nums、>50 行分页）；正文 12–14px；
命令面板 ⌘K 跳转；目标尺寸 ≥24×24px。

## Spacing / Radius / Elevation

间距 4px 基准阶梯（4/8/12/16/24）；圆角：组件 6px、卡片 8px、弹层 8px；
阴影：暗色下卡片只用 1px border 无阴影，亮色用 shadow-sm（`0 1px 2px rgb(0 0 0/.05)`）。

## Motion（Subtle 档）

- 只动画 transform/opacity；禁 width/height/top/left；时长 150–200ms，列表 stagger 30–50ms/项。
- 尊重 `prefers-reduced-motion`（全量回退为无动画）。
- 每屏最多 1–2 个动效元素；按压反馈 scale 0.97。

## Charts（`charts.csv` 规范落地）

- 网格线低对比（暗色 gray-800 级）；图例可点击切换序列；禁纯颜色编码（线型/标注辅助）。
- ≥1000 点降采样；折线 <1000 点 SVG（Recharts）；KPI 用紧凑表格而非大数字卡片（数据密度优先）。
- 图表必须有加载态与空态；数值精度统一 3–4 位有效数字。

## Accessibility（验收线）

正文对比度 ≥4.5:1（大字/图形 ≥3:1）；focus-visible 2px ring（--ring）；
图标按钮必须 aria-label；图表数据回退为可读的表格/文字。

## Avoid

 ornate 装饰、营销式 hero、无筛选的大表格、自动播放影像（影像回放必须由用户触发）、
 亮色 3D 视口默认（默认 dark）、硬编码数据集名/颜色值进组件。

## 页面路由（信息架构）

| 路由 | 页面 | 核心内容 |
|---|---|---|
| `/` | Overview | 数据集卡片（platform 分组）+ 指标速览 |
| `/explore/:dataset/:split` | Explorer | 帧 A/B 模型对比、帧间差分、叠加、ROI、GT/控制点层 |
| `/scene/:dataset/:split` | Scene | 融合点云流式加载、俯视轨迹、fly-through 回放 |
| `/controls/:dataset/:split` | Controls | 稀疏控制点 2D 诊断（多源/着色/悬停/播放） |
| `/metrics` | Metrics | 模型×指标矩阵、per-frame 误差曲线、summary 对比 |
| `/tasks` | Tasks | 任务列表、新建向导、WS 实时日志 |

## 平台扩展（vehicle → uav → satellite）

所有数据集/split/相机/控制点/模型下拉框必须来自 `/api/registry`（禁止硬编码）。
`platform` 字段驱动差异面板；新增平台 = 注册表配置 + （可选）一个按 platform 懒加载的面板组件。
