# Nanling Biodiversity Vegetation Function Gain Assessment System

## 中文简介

`Nanling_Biogain` 是一个面向南岭区域的 Windows 桌面研究软件，用于评估生物多样性提升对植被功能的增益，而不只是做一般的预测建模。

当前重点回答 3 类问题：
- 树种多样性 `tree_diversity` 每增加 1 个单位，`GPP`、`LAI`、`VOD` 会增加多少
- 结构多样性 `structure_diversity` 每增加 1 个单位，`GPP`、`LAI`、`VOD` 会增加多少
- 在控制气候和地形变量后，生物多样性变量是否仍然具有独立贡献

当前版本已经完成一条可用主链：
- 手动加载或加载多年平均预处理数据
- 栅格对齐检查与必要重采样
- 像元级样本表构建
- `RandomForestRegressor` 基线训练
- 生物多样性 `+1` 增益评估
- 绝对增益和百分比增益预览
- 中英文界面切换

## English Overview

`Nanling_Biogain` is a Windows-first desktop research application for the Nanling region. It focuses on biodiversity-driven gain in vegetation functions rather than generic prediction alone.

The current scientific questions are:
- How much do `GPP`, `LAI`, and `VOD` increase when `tree_diversity` increases by one unit
- How much do `GPP`, `LAI`, and `VOD` increase when `structure_diversity` increases by one unit
- Whether biodiversity variables still contribute after climate and terrain are controlled

The current version already supports a usable end-to-end workflow:
- Manual raster loading or prepared mean-dataset loading
- Grid alignment checks and required resampling
- Pixel-level sample-table building
- `RandomForestRegressor` baseline training
- Biodiversity `+1` gain assessment
- Absolute and percent gain previews
- Chinese / English UI switching

## 当前目录 / Repository Structure

- `main.py`: 桌面应用入口 / desktop entry point
- `ui/`: 界面、后台任务和多语言文本 / UI, background tasks, translations
- `core/`: 栅格处理、建模、情景评估 / raster processing, modeling, scenarios
- `utils/`: 常量、路径、导出辅助工具 / constants, paths, export helpers
- `docs/`: 需求、技术路线、架构说明 / scope, stack, architecture notes

## 运行环境 / Environment

推荐环境：
- Python `3.11` 作为目标版本
- 当前开发与测试实际主要在 `conda` 环境 `da` 中完成

核心依赖：
- `PySide6`
- `rasterio`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `openpyxl`
- `python-docx`

## 快速开始 / Quick Start

如果使用现有环境：

```bash
conda activate da
python main.py
```

如果按环境文件新建：

```bash
conda env create -f environment.yml
conda activate nanling-biogain
python main.py
```

## 典型流程 / Typical Workflow

1. 打开软件，加载南岭研究区相关栅格  
   Open the app and load Nanling study-area rasters.

2. 确认响应变量至少有一个：`GPP` / `LAI` / `VOD`，并同时提供  
   `tree_diversity` 和 `structure_diversity`  
   Ensure at least one response raster is available and both biodiversity rasters are provided.

3. 软件会检查栅格是否对齐；若未对齐，会按响应变量网格重采样  
   The app checks alignment and resamples to the response grid when needed.

4. 构建样本表并训练随机森林  
   Build the sample table and train the random forest baseline.

5. 在“生物多样性增益评估”窗口中运行 `+1` 情景  
   Run the `+1` scenario in the Biodiversity Gain Assessment window.

6. 查看报告、增益地图和预测结果  
   Review the report, gain maps, and predictor results.

## 当前基线 / Current Baseline

- 模型 / Model: `RandomForestRegressor`
- 响应变量 / Responses: `GPP`, `LAI`, `VOD`
- 主要解释变量 / Main predictors:
  - `tree_diversity`
  - `structure_diversity`
  - `MAT`
  - `MAP`
  - `VPD`
  - `SM`
  - `SSRD`
  - `DEM`
  - `slope`
  - `aspect`

## 数据与提交说明 / Data And Git Notes

首次代码提交默认**不包含本地数据与输出结果**：
- `data/`
- `outputs/`

也就是说，这个仓库优先提交：
- 源代码
- 文档
- 环境配置

而不提交本地预处理栅格、模型结果和临时输出。

The initial code commit is intended to exclude local data and generated outputs. The repository should primarily track source code, docs, and environment configuration.

## 后续方向 / Next Steps

- 报告导出进一步规范化 / cleaner report export
- 更丰富的增益图表达 / richer gain-map presentation
- SHAP / PDP 等解释分析 / explainability workflows
- Windows 打包发布 / Windows packaging and release
