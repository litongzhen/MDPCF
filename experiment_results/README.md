# 实验结果文件说明

本目录整理五个数据集上的已有实验结果，包含主方法、两种表示学习消融和五种 baseline。随机种子为 42、43、44，黑盒分类器固定为 seed 42。

## 在新环境中运行实验

以下命令均在项目根目录 `MDPCF/` 执行，而不是在 `experiment_results/` 内执行。完整批量入口使用 PowerShell；安装及 Python 单步命令也可在其他操作系统使用。

### 1. 下载代码并创建环境

```powershell
git clone https://github.com/litongzhen/MDPCF.git
cd MDPCF
conda create -n mdpcf python=3.10 -y
conda activate mdpcf
```

也可使用本机 Python 3.10 创建虚拟环境，替代上面的 Conda 命令：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安装依赖与 baseline 源码

DiCoFlex 上游源码需单独取得，目录必须为 `baselines/DiCoFlex-main`。使用前确认其上游使用条款。

```powershell
git clone https://github.com/ofurman/DiCoFlex.git baselines/DiCoFlex-main
powershell -ExecutionPolicy Bypass -File scripts/setup_environment.ps1 -TorchFlavor cpu -Baselines
python scripts/check_environment.py --baselines
```

若使用 NVIDIA GPU，且驱动支持 CUDA 13.0 对应的 PyTorch wheel，将安装命令中的 `-TorchFlavor cpu` 改为 `-TorchFlavor cu130`。CPU 可以运行，但正式批量实验耗时较长。

上述安装脚本会安装 `requirements-baselines.txt`，并用 `--no-deps` 安装仓库内的 DiCE 和 CARLA。无需另外安装旧版 CARLA 指定的 Torch/TensorFlow。依赖版本见项目根目录的 `requirements.txt` 与 `requirements-baselines.txt`；它们不是全部传递依赖的锁文件。

### 3. 准备数据

Adult 可自动下载并预处理：

```powershell
python download_dataset.py
python process_dataset.py --dataname adult
```

其余原始数据需自行取得，并按 `data/Info/` 中对应 JSON 的格式和路径放置：

| 数据集 | 原始文件位置 |
|---|---|
| Bank | `data/bank/bank.csv` |
| Default | `data/default/default of credit card clients.xls` |
| GMC | `data/gmc/cs-training.csv` |
| Lending Club | `data/lending-club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv` |

```powershell
python process_dataset.py --dataname bank
python process_dataset.py --dataname default
python process_dataset.py --dataname gmc
python process_dataset.py --dataname lending-club
```

预处理后各数据集目录应有 `info.json`、`train.csv`、`test.csv` 和对应的 NumPy 数据文件。数据及模型不随 Git 仓库上传。

### 4. 先验证最小流程

准备好 Adult 后，可先用少量样本检查安装和流程：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_quickstart.ps1 -Python python -Gpu -1 -Epochs 20 -NumSamples 3
```

CUDA 环境下可将 `-Gpu -1` 改为 `-Gpu 0`。该命令仅验证流程，使用独立输出目录，不对应本目录的正式结果。

### 5. 主方法与两种消融

```powershell
./scripts/run_final_main_ablations.ps1 -Python python -Resume
```

默认运行五个数据集、seed 42/43/44、三种表示模型，共 45 项。脚本先训练或读取固定 seed 42 的黑盒，再训练潜模型、生成和评估反事实。默认潜模型训练 2000 epoch，每个源请求 10 个输出，候选数 50，生成最多 3000 步；各数据集的具体参数由脚本定义。

若先运行一个完整的小项：

```powershell
./scripts/run_final_main_ablations.ps1 -Python python -Datasets adult -Seeds 42 -Variants csvae_transformer -Resume
```

### 6. 五种 baseline

先完成主方法脚本对对应数据集的黑盒准备，再运行：

```powershell
./scripts/run_final_baselines.ps1 -Python python -Resume
```

默认运行五个数据集、五种 baseline、seed 42/43/44，共 75 项。只运行 Adult 的 DiCoFlex / seed 42 时：

```powershell
./scripts/run_final_baselines.ps1 -Python python -Datasets adult -Methods dicoflex -Seeds 42 -Resume
```

各 baseline 使用其适配器中实际定义的搜索预算；通用命令中的 `max_iter` 不一定控制每种 baseline 的内部迭代次数。`-Resume` 跳过已完成评估的项目，不表示所有 baseline 都支持生成过程中的逐源恢复。

### 7. 查找输出与恢复

- 主方法及消融的逐项输出：`results/final_main_ablation_3000/runs/`。
- 主方法及消融汇总：`results/final_main_ablation_3000/per_seed.csv`、`mean_std.csv`。
- baseline 逐项输出：`results/final_baselines_multiseed_3000/runs/`。
- baseline 汇总：`results/final_baselines_multiseed_3000/all_methods.csv`。
- 运行日志：`logs/final_main_ablation/`、`logs/final_baselines/`。

每个结果子目录内的 `cfs.csv` 保存生成样本，`original_test_samples.csv` 保存源样本，`config.json` 记录参数，`evaluation_summary.csv` 保存评价指标。重复执行相同批量命令并保留 `-Resume` 即可继续未完成项目。更换参数时使用独立输出目录和 run tag，不要覆盖已有实验。

本目录的 CSV 是已保存结果的整理，不会随重新运行自动覆盖。新环境的具体数值以实际输出为准；这里只提供执行入口，不承诺逐位复现已有数值，也不表示旧 CARLA 实现的已知限制已被修复。逐项来源见 `sources.json`。

## 文件内容

| 文件 | 内容 |
|---|---|
| `results.md` | 可直接阅读的结果表，显示各项指标的均值和标准差。 |
| `mean_std.csv` | 全部方法的汇总结果，共 40 组（5 个数据集 × 8 种方法/变体）。以 `_mean` 结尾的列是均值，`_std` 是样本标准差。 |
| `per_seed.csv` | 全部方法的逐 seed 结果，共 120 行（5 个数据集 × 8 种方法/变体 × 3 个 seed）。 |
| `all_methods.csv` | 五种 baseline 的完整指标表，共 75 行，保留原始总表的列结构，包括计数、细分指标和结果目录。此文件不包含主方法及两种消融。 |
| `metric_counts.csv` | 每个数据集、方法、指标实际参与统计的 seed 数量。未定义的指标不计入数量；此处不是反事实样本数量。 |
| `sources.json` | 每组结果的原始存储位置，以及源汇总文件的 SHA256，便于追溯数据来源。 |
| `main_ablations/per_seed.csv` | 原主方法及两种消融的逐 seed 汇总，共 45 行，保留原文件格式。 |
| `main_ablations/mean_std.csv` | 原主方法及两种消融的均值和标准差，共 15 组。 |

## 方法名称

- `csvae_transformer`：MDPCE 主方法，Transformer + CSVAE。
- `csvae_mlp`：用 MLP 替换 Transformer 的骨干消融。
- `standard_vae`：使用标准 VAE 的表示学习消融。
- `dice`、`dicoflex`、`cchvae`、`revise`、`wachter`：五种 baseline。

## 主要字段

| 字段 | 含义 |
|---|---|
| `dataset` | 数据集名称：Adult、Bank、Default、GMC、Lending Club。 |
| `method` / `variant` | 方法或消融变体名称。 |
| `seed` | 实验随机种子。 |
| `black_box_seed` | 黑盒分类器种子。 |
| `coverage` | 至少返回一个有效反事实的源样本比例，越高越好。 |
| `validity_generated` | 已返回反事实中的有效比例，越高越好。 |
| `validity_expected` | 有效反事实数除以请求总数，即 Valid-CF Yield，越高越好。每个源请求 10 个输出，单输出方法的该指标上限为 0.1。 |
| `mean_target_prob` | 有效反事实的平均目标类别预测概率。 |
| `diversity` | 同一源样本对应的有效反事实集合的多样性，越高越好；不足两个有效输出时无法定义。 |
| `proximity` | 有效反事实相对原始实例的平均特征变化距离，越低越好。 |
| `lof_log_scale` | 有效反事实的平均 LOF 分数，越低表示相对训练数据的局部密度一致性越好；字段名沿用代码，保存值不是对数变换后的 LOF。 |

CSV 中的空值及阅读表中的 `--` 表示指标未定义，不表示数值为零。均值、标准差应结合 `metric_counts.csv` 阅读；标准差采用样本标准差。
