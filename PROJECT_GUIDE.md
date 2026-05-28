# MDPCF 项目说明

## 这个项目现在做了什么

这个仓库的主线已经从原来的 VAE 改成了 CSVAE。

当前 MDPCF 主流程是：

1. 训练 CSVAE
2. 把输入编码成潜变量 `z` 和 `w`
3. 生成反事实时固定 `w = w0`
4. 只在 `z` 空间里优化
5. 用多方向初始化 + 多样性损失，让多个反事实尽量分散

baseline 还保留着，不改它们的逻辑。

## 目录用途

### 根目录

- `main.py`：统一入口
- `utils.py`：参数定义、方法分发
- `utils_train.py`：数据预处理、训练参数命名
- `download_dataset.py`：下载数据集
- `process_dataset.py`：数据预处理
- `evaluation_framework/evaluate.py`：统一评估
- `PROJECT_GUIDE.md`：这份说明
- `README.md`：原始英文说明

### `mdpcf/`

- `mdpcf/vae/model.py`：CSVAE、编码器、解码器、黑盒分类器
- `mdpcf/vae/main.py`：检查黑盒分类器是否存在，并训练 CSVAE
- `mdpcf/vae/train_black_box.py`：单独训练黑盒分类器
- `mdpcf/vae/test_black_box.py`：测试已训练的黑盒分类器
- `mdpcf/vae/run_black_box.py`：黑盒分类器训练/检查后立即测试
- `mdpcf/latent_utils.py`：潜变量加载、解码、等角方向生成
- `mdpcf/sample.py`：MDPCF 反事实生成主流程

### `baselines/`

各 baseline 方法的实现：

- `baselines/dice/`
- `baselines/revise/`
- `baselines/cchvae/`
- `baselines/wachter/`
- `baselines/CARLA/`

### `data/Info/`

每个数据集的元信息：

- 列名
- 目标列
- 类别列 / 数值列
- 正负类名称
- 编码和逆编码信息

### `CSVAE/`

一套独立的 CSVAE 参考实现/实验代码。

### `counterfactual_results*/`

生成的反事实输出目录，默认不进 Git。

### `figures/`

实验图和可视化结果。

## 怎么训练

先准备环境：

```bash
conda create -n mdpcf python=3.10
conda activate mdpcf
pip install -r requirements.txt
```

准备数据：

```bash
python download_dataset.py
python process_dataset.py
```

训练 CSVAE：

```bash
python main.py --dataname adult --method vae --mode train --gpu 0
```

这一步现在只会检查黑盒分类器是否已经存在，不会重新训练黑盒。

只训练并检查黑盒分类器：

```bash
python main.py --dataname adult --method black_box --mode run --gpu 0 --hidden_dims 16
```

常用参数：

- `--vae_epochs`：CSVAE 训练轮数
- `--black_box_epochs`：黑盒分类器训练轮数
- `--gpu -1`：强制 CPU
- `--gpu 0`：用第 0 张 GPU

## 怎么启动生成反事实

MDPCF：

```bash
python main.py --dataname adult --method mdpcf --mode sample --gpu 0 --num_samples 2 --total_CFs 4
```

常用采样参数：

- `--num_samples`：选多少个测试样本
- `--total_CFs`：每个样本生成多少个反事实
- `--min_iter` / `--max_iter`：优化步数
- `--proximity_weight_input`
- `--proximity_weight_latent`
- `--diversity_weight`
- `--latent_init_radius`

baseline 也用同一个入口，只改 `--method`：

```bash
python main.py --dataname adult --method wachter --mode sample --gpu 0 --num_samples 2
```

## 评估怎么跑

```bash
python main.py --dataname adult --method mdpcf --mode evaluate --num_samples 2 --total_CFs 4
```

评估会读取对应的 `counterfactual_results/.../cfs.csv`，计算：

- validity
- sparsity
- proximity
- outlier / density 类指标

## 当前 MDPCF 的实现细节

### 1. CSVAE 替换

`mdpcf/vae/model.py` 里现在主链路是 CSVAE。

- `z`：和 label 相关的潜变量
- `w`：保留样本其他因素

解码时用 `(z, w)` 一起重建输入。

### 2. 固定 `w`

反事实生成时固定原样本的 `w0`，只优化 `z`。

这让反事实变化更集中在决策相关因素上。

### 3. 多样性采样

初始化阶段会先构造多个潜空间方向，尽量让它们分散。

如果维度足够，会用等角方向。
如果维度不够，会退化成确定性候选 + 最远点贪心筛选。

然后在优化里加入多样性损失，鼓励多个反事实彼此远离。

## 输出在哪里

默认结果保存在：

```text
counterfactual_results/<dataname>/...
```

每次采样通常会输出：

- `cfs.csv`
- `original_test_samples.csv`

## 一个推荐的最小流程

```bash
python download_dataset.py
python process_dataset.py
python main.py --dataname adult --method vae --mode train --gpu 0
python main.py --dataname adult --method mdpcf --mode sample --gpu 0 --num_samples 2 --total_CFs 4
python main.py --dataname adult --method mdpcf --mode evaluate --gpu 0 --num_samples 2 --total_CFs 4
```
