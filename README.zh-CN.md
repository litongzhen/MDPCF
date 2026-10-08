# MDPCE 中文运行说明

仓库名和命令保留 MDPCF；论文方法名为 MDPCE。主方法使用 Transformer + CSVAE，固定源样本 z0，在标签相关 w 分支搜索多个反事实。论文草稿、原始数据和预训练模型不包含在这次代码发布中。

## 1. 获取代码与安装环境

```powershell
git clone git@github.com:litongzhen/MDPCF.git
cd MDPCF
conda create -n mdpcf python=3.10
conda activate mdpcf
powershell -ExecutionPolicy Bypass -File scripts/setup_environment.ps1 -TorchFlavor cpu
```

SSH 克隆要求先配置 GitHub SSH；也可以用 HTTPS 地址克隆。本地已有可运行环境时不要重新创建。安装脚本只操作当前选定的 Python 环境，不创建 `.venv`。

有兼容 CUDA 13.0 的 NVIDIA 驱动时，将 `cpu` 换成 `cu130`。核心依赖有版本固定，但仍未验证全新隔离安装，不能保证所有系统一键成功或跨硬件逐位一致。安装后执行 `python scripts/check_environment.py` 检查导入。

## 2. 准备 Adult 数据

在仓库根目录执行：

```powershell
python download_dataset.py
python process_dataset.py --dataname adult
```

当前下载脚本只自动处理 Adult。其余四个数据集的文件位置见英文 README 的 Data 表和 `data/Info/*.json`，放入原文件后用相应 `--dataname` 预处理。

## 3. 完整运行主方法

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_quickstart.ps1 -Python python -Gpu -1
```

该脚本按顺序完成：黑盒分类器训练/检查 → Transformer CSVAE 训练/复用 → 3 个源样本的反事实生成 → 共享指标评估。使用 seed=42、固定黑盒 seed=42、K=10、M=50、2000 训练 epoch，生成最多 3000 轮；输出独立保存在 `results/quickstart/`。

GPU 环境用 `-Gpu 0`。安装流程检查可以用 `-Epochs 20`，但其效果不能作为论文实验结果。脚本自动按训练 epoch 区分模型 run tag、按 epoch/源样本数区分输出目录，避免 20 轮检查覆盖 2000 轮模型；默认输出为 `results/quickstart/e2000_n3/`。修改其他协议参数时仍需另设 run tag/输出目录，正式实验使用下一节的 runner。

训练模型并未随仓库发布，首次运行需要训练，因此不是瞬间生成结果。英文 README 的 Minimal Run 给出了每个阶段分别执行的 Python 命令，也适用于非 Windows 环境。

## 4. 查结果和恢复

在 `results/quickstart/` 下递归查找：

- `cfs.csv`：返回的反事实及对应源 id。
- `original_test_samples.csv`：原始源样本。
- `config.json`：实际运行设置。
- `evaluation_summary.csv`：coverage、validity、target probability、diversity、proximity、LOF 等；`validity_expected` 列表示 valid-CF yield。

每个完成的源样本有 checkpoint。中断后重新运行相同命令会复用已完成源；当前未完成源可能重跑。不要删除结果或在同目录改变协议。模型文件在对应模型/checkpoint 目录，不上传 Git。

## 5. 论文正式主线与消融

准备好全部五个数据集后：

```powershell
./scripts/run_final_main_ablations.ps1 -Python python -Resume
```

这是 5 数据集 × 3 表示变体 × 3 seed，共 45 项，不是快速测试。变体为 CSVAE-Transformer、参数量匹配的 CSVAE-MLP、Transformer-standard-VAE；固定黑盒 seed=42，正式源为 100 个共同 eligible sources，K=10、M=50、max_iter=3000。Adult/Bank/Lending Club 目标类为 1，Default/GMC 为 0。各数据集参数由 runner 定义。

正式 runner 及其汇总脚本保留在仓库中，运行后会生成指标汇总和配置记录。已有结果、环境快照、论文材料、原始数据及预训练模型不上传，需要在本地准备数据并运行生成。

## 6. 可选 baseline

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_environment.ps1 -TorchFlavor cpu -Baselines
```

DiCE/CARLA 的适配源码与许可证包含在仓库中；安装时使用 `--no-deps` 避免它们旧版依赖覆盖当前环境。DiCoFlex 上游源码授权尚未确认，因此不打包，需从英文 README 的官方链接另行获取到 `baselines/DiCoFlex-main/`。在此之前安装脚本只检查其他 baseline。

完整 baseline 队列需要先完成固定黑盒准备并放好 DiCoFlex 源码：

```powershell
./scripts/run_final_baselines.ps1 -Python python -Resume
```

## 7. 常见问题

- 找不到数据：检查原文件位置、文件名和预处理是否完成。
- 找不到黑盒：先训练黑盒，并保持同一 dataset/run tag/分类器参数。
- 导入失败：确认 Python 3.10 和选定环境，执行环境检查脚本查看缺失模块。
- 跑得慢：首次会训练模型；CPU 慢于 GPU，三 seed 全量实验是长任务。
- 想测试代码：安装 `requirements-dev.txt` 后运行 `python -m pytest tests -q`；不要对整个 vendored 仓库做测试发现。
