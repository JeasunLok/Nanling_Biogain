# Assessment System for Biodiversity Impacts on Ecosystem Service Functions

## 中文

### 项目简介

“生物多样性对生态系统服务功能影响评估系统”是一个面向区域生态分析的桌面软件。项目以 Python 和 PySide6 为基础，围绕生物多样性对生态系统服务功能的影响开展像元级建模、情景分析与结果展示。

本系统关注以下核心问题：
- 树种多样性 `tree_diversity` 增加时，`GPP`、`LAI`、`VOD` 的变化幅度
- 结构多样性 `structure_diversity` 增加时，`GPP`、`LAI`、`VOD` 的变化幅度
- 在控制气候与地形变量后，生物多样性变量是否仍具有独立贡献

### 当前功能

当前版本已实现以下主要功能：
- 多源栅格手动加载与多年平均数据加载
- 栅格对齐检查与基于响应变量网格的必要重采样
- 南岭区域裁剪预览与矢量边界叠加
- 像元级样本表构建
- `RandomForestRegressor` 基线模型训练
- 生物多样性情景增益评估
- 绝对增益与百分比增益可视化
- 中英文界面切换

### 技术栈

- Python
- PySide6
- rasterio
- numpy
- pandas
- scikit-learn
- matplotlib
- openpyxl
- python-docx

### 项目结构

- `main.py`：桌面应用入口
- `ui/`：界面、交互、多语言与后台任务
- `core/`：栅格处理、样本表构建、建模与情景分析
- `utils/`：常量、路径、导出与通用辅助函数
- `docs/`：范围定义、技术路线与架构说明

### 运行方式

当前支持两种运行方式：

1. 面向正式使用的 Windows 安装包  
安装完成后，用户可直接通过桌面或开始菜单启动软件，无需手动配置 Python 或 Conda 环境。

2. 面向开发与科研复现的 Conda 环境  
如需在本地继续开发、调试或复现实验流程，可使用环境配置文件创建运行环境。

相关环境配置保留在：
- `environment.yml`
- `requirements.txt`

示例：

```bash
conda env create -f environment.yml
conda activate nanling-biogain
python main.py
```

### 分析流程

1. 加载响应变量栅格与解释变量栅格  
2. 检查对齐关系并完成必要重采样  
3. 构建样本表  
4. 训练随机森林模型  
5. 进入“生物多样性增益评估”窗口运行 `+1` 情景  
6. 查看报告、增益地图与预测结果  

### 数据说明

仓库默认不包含本地数据、预处理结果与运行输出。  
如需获取数据，请联系：

`luojsh7@mail2.sysu.edu.cn`

### 许可

本仓库源代码采用 `Apache License 2.0`。

---

## English

### Overview

The Assessment System for Biodiversity Impacts on Ecosystem Service Functions is a desktop application for regional ecological analysis. Built with Python and PySide6, it supports pixel-level modeling, scenario analysis, and visualization of biodiversity effects on ecosystem service functions.

The system is designed to address the following questions:
- How `GPP`, `LAI`, and `VOD` change when `tree_diversity` increases
- How `GPP`, `LAI`, and `VOD` change when `structure_diversity` increases
- Whether biodiversity variables retain independent contributions after controlling for climate and terrain

### Implemented Features

The current release includes:
- Manual raster loading and prepared mean-dataset loading
- Grid alignment checks and response-grid-based resampling when required
- Nanling-only preview with vector boundary overlay
- Pixel-level sample-table construction
- `RandomForestRegressor` baseline model training
- Biodiversity gain assessment
- Absolute-gain and percent-gain visualization
- Chinese / English UI switching

### Technology Stack

- Python
- PySide6
- rasterio
- numpy
- pandas
- scikit-learn
- matplotlib
- openpyxl
- python-docx

### Repository Structure

- `main.py`: desktop application entry point
- `ui/`: UI, interaction, translations, and background tasks
- `core/`: raster processing, sample-table construction, modeling, and scenarios
- `utils/`: constants, paths, exports, and shared helpers
- `docs/`: scope, technical notes, and architecture documents

### Running The Application

Two usage modes are supported:

1. Windows installer for end users  
After installation, the application can be launched directly from the desktop or the Start menu without manual Python or Conda setup.

2. Conda environment for development and research reproduction  
For local development, debugging, or workflow reproduction, the project can also be run from a Conda environment.

Environment configuration files are retained in:
- `environment.yml`
- `requirements.txt`

Example:

```bash
conda env create -f environment.yml
conda activate nanling-biogain
python main.py
```

### Analysis Workflow

1. Load response rasters and predictor rasters  
2. Check alignment and complete required resampling  
3. Build the sample table  
4. Train the random forest model  
5. Open the Biodiversity Gain Assessment window and run the `+1` scenario  
6. Review reports, gain maps, and prediction results  

### Data Access

The repository does not include local datasets, prepared rasters, or runtime outputs by default.  
For data access, please contact:

`luojsh7@mail2.sysu.edu.cn`

### License

The source code in this repository is licensed under the `Apache License 2.0`.
