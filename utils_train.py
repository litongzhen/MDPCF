import numpy as np
import json
import os
import torch
import src
from torch.utils.data import Dataset

def get_name_form_args(args):
    """
    根据训练参数生成较短的模型文件名，用于保存和加载模型

    参数:
        args: 命令行参数对象，包含所有训练配置

    返回:
        str: 由核心参数拼接而成的短文件名字符串

    示例输出:
        "num[minmax]_loss[L2]_csvae_tokdec_gs[t1.0]_cont[sigmoid]"
    """
    short_num_encoding = args.num_encoding.replace("_torch", "").replace(
        "min_max", "minmax"
    )
    latent_model = getattr(args, "latent_model", "csvae")
    csvae_backbone = getattr(args, "csvae_backbone", "transformer")
    opts_csvae = [
        f"num[{short_num_encoding}]",
        f"loss[{args.num_reconstr_loss}]",
        latent_model,
        "tokdec",
    ]
    if csvae_backbone != "transformer":
        opts_csvae.append(f"bb[{csvae_backbone}]")

    if args.kl_weight > 0:
        opts_csvae.append(f"kl[{args.kl_weight}]")

    if args.gumbel_softmax:
        opts_csvae.append(f"gs[t{args.tau}]")

    opts_csvae.append("cont[sigmoid]" if args.sigmoid else "cont[linear]")

    token_dim = int(getattr(args, "token_dim", 4))
    latent_dim = int(getattr(args, "latent_dim", 4))
    if (token_dim, latent_dim) != (4, 4):
        opts_csvae.append(f"dim[t{token_dim}_z{latent_dim}]")

    run_tag = getattr(args, "run_tag", "")
    if run_tag:
        opts_csvae.extend([f"run[{run_tag}]", f"seed[{args.seed}]"])

    return "_".join(opts_csvae)


def save_run_config(args, output_dir, filename="config.json"):
    """
    将运行参数落盘，方便复现实验。
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, filename)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False, sort_keys=True)
    return config_path


def get_black_box_model_path(
    args=None, data_dir="", cat_encoding=None, hidden_dims=None, gumbel_softmax=None
):
    """
    统一黑盒分类器文件名。
    """
    if args is not None:
        hidden_dims = getattr(args, "hidden_dims", hidden_dims)
        gumbel_softmax = getattr(args, "gumbel_softmax", gumbel_softmax)

    if hidden_dims is not None:
        base_name = f"black_box_mlp_hidden_{hidden_dims}"
        run_tag = getattr(args, "run_tag", "") if args is not None else ""
        if run_tag:
            black_box_seed = getattr(args, "black_box_seed", None)
            artifact_seed = args.seed if black_box_seed is None else black_box_seed
            base_name = f"{base_name}_run[{run_tag}]_seed[{artifact_seed}]"
        return os.path.join(data_dir, f"{base_name}.pkl")
    if cat_encoding == "one-hot-drop-bin":
        return os.path.join(data_dir, "black_box_mlp_ohe_drop_bin.pkl")
    if gumbel_softmax is False:
        return os.path.join(data_dir, "black_box_mlp_no_gumbel_softmax.pkl")
    return os.path.join(data_dir, "black_box_mlp.pkl")

def get_optim_params_form_args(args):
    """
    生成反事实解释优化参数的字符串描述，用于保存优化配置和结果

    参数:
        args: 命令行参数对象，包含反事实解释的优化配置

    返回:
        str: 由优化参数拼接而成的配置字符串

    反事实解释的关键参数说明:
        - proximity_weight_input: 输入空间邻近性权重（原始样本与反事实样本的接近程度）
        - diversity_weight: 多样性权重（多个反事实样本之间的差异性）
        - latent_init_radius: 潜在空间初始化半径
        - proximity_weight_latent: 潜在空间邻近性权重
        - plausibility_weight_latent: 合理性权重（反事实样本的可信度）
        - immutable: 不可变特征（如年龄、种族等不应改变的特征）
    """
    run_tag = getattr(args, "run_tag", "")
    if run_tag:
        # Seeded sweeps retain the full configuration in config.json and use a
        # short directory component to remain below Windows path-length limits.
        return f"run[{run_tag}]_seed[{args.seed}]"

    # 基础参数：输入空间邻近性权重
    optimization_parameters = f"prox_x_w[{args.proximity_weight_input}]"

    # 可选参数：多样性权重（用于生成多样化的反事实解释）
    if hasattr(args, "diversity_weight"):
        optimization_parameters = (
            f"{optimization_parameters}_div_w[{args.diversity_weight}]"
        )

    if getattr(args, "diversity_loss", "average_distance") != "average_distance":
        optimization_parameters = (
            f"{optimization_parameters}_div_loss[{args.diversity_loss}]"
        )

    # 可选参数：潜在空间初始化半径
    if hasattr(args, "latent_init_radius"):
        latent_radius_strategy = getattr(args, "latent_init_radius_strategy", "fixed")
        if latent_radius_strategy == "auto_nn":
            optimization_parameters = (
                f"{optimization_parameters}_zinit[auto_nn]"
                f"_zscale[{getattr(args, 'latent_init_radius_scale', 1.0)}]"
            )
        else:
            optimization_parameters = (
                f"{optimization_parameters}_zinit[{args.latent_init_radius}]"
            )

    # 可选参数：隐藏层维度
    if args.hidden_dims is not None:
        optimization_parameters = f"{optimization_parameters}_h[{args.hidden_dims}]"

    # 非DiCE方法的特有参数
    if args.method != "dice":

        # 潜在空间邻近性权重和损失类型
        if args.proximity_weight_latent > 0:
            optimization_parameters = f"{optimization_parameters}_prox_z_w[{args.proximity_weight_latent}]_prox_z_loss[{args.proximity_latent_loss}]"

        # 合理性权重和损失类型（确保反事实样本在数据分布内）
        if args.plausibility_weight_latent > 0:
            optimization_parameters = f"{optimization_parameters}_plaus_z_w[{args.plausibility_weight_latent}]_plaus_z_loss[{args.plausibility_latent_loss}]"

        # 输入空间邻近性损失类型
        if args.proximity_input_loss is not None:
            optimization_parameters = f"{optimization_parameters}_prox_x_loss[{args.proximity_input_loss}]"

        if getattr(args, "validity_loss", "cross_entropy") != "cross_entropy":
            optimization_parameters = f"{optimization_parameters}_valid_loss[{args.validity_loss}]"

        if getattr(args, "validity_margin", 1.0) != 1.0:
            optimization_parameters = f"{optimization_parameters}_valid_m[{args.validity_margin}]"

        if getattr(args, "categorical_straight_through", False):
            optimization_parameters = f"{optimization_parameters}_cat_st[{args.categorical_straight_through}]"

        if getattr(args, "posthoc_sparsity", False):
            optimization_parameters = f"{optimization_parameters}_posthoc_sp[{args.posthoc_sparsity}]"

        if getattr(args, "posthoc_diversity", False):
            optimization_parameters = f"{optimization_parameters}_posthoc_div[{args.posthoc_diversity}]"

        candidate_CFs = getattr(args, "candidate_CFs", 0)
        if candidate_CFs and candidate_CFs != getattr(args, "total_CFs", candidate_CFs):
            optimization_parameters = f"{optimization_parameters}_candCFs[{candidate_CFs}]"

        if getattr(args, "cf_selection_metric", "mixed_distance") != "mixed_distance":
            optimization_parameters = f"{optimization_parameters}_select[{args.cf_selection_metric}]"

        if getattr(args, "trajectory_quality_selection", False):
            optimization_parameters = f"{optimization_parameters}_trajq"

        if getattr(args, "enforce_constraints", False):
            optimization_parameters = f"{optimization_parameters}_constraints"

        # 不可变特征配置（这些特征在反事实解释中不应改变）
        if args.immutable:
            optimization_parameters = f"{optimization_parameters}_immutable[{args.immutable}]"

    return optimization_parameters

class TabularDataset(Dataset):
    """
    表格数据集类，继承自PyTorch的Dataset，用于DataLoader加载数据

    属性:
        X_num (numpy.ndarray/tensor): 数值特征矩阵 [n_samples, n_numerical_features]
        X_cat (numpy.ndarray/tensor): 分类特征矩阵 [n_samples, n_categorical_features]，可选
        y (torch.Tensor): 二分类标签 [n_samples]，0或1

    使用示例:
        dataset = TabularDataset(X_num_train, X_cat_train, y_train, info)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        for batch_num, batch_cat, batch_y in dataloader:
            # 训练模型...
    """

    def __init__(self, X_num, X_cat=None, y=None, info=None):
        """
        初始化表格数据集

        参数:
            X_num (numpy.ndarray): 数值特征，形状 [n_samples, n_numerical_features]
            X_cat (numpy.ndarray, optional): 分类特征，形状 [n_samples, n_categorical_features]
            y (numpy.ndarray, optional): 标签数组，形状 [n_samples, 1]，包含类别名称
            info (dict, optional): 数据集信息字典，必须包含:
                - "target_class": 正类名称（映射为1）
                - "negative_class": 负类名称（映射为0）
        """
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = None

        # 如果提供了标签，进行类别映射
        if y is not None:
            # 创建类别名称到0/1的映射字典
            # 例如: {"收入>50K": 1, "收入<=50K": 0}
            class_mapping = {info["target_class"]: 1, info["negative_class"]: 0}

            # 将类别名称转换为0/1标签
            # y的形状为 [n_samples, 1]，所以取 class_name[0]
            y_n = np.array([class_mapping[class_name[0]] for class_name in y])

            # 转换为PyTorch tensor
            self.y = torch.tensor(y_n)

    def __getitem__(self, index):
        """
        获取单个样本

        参数:
            index (int): 样本索引

        返回:
            tuple: (数值特征, 分类特征, 标签) 或 (数值特征, 分类特征)
                - 有标签时返回三元组
                - 无标签时返回二元组
        """
        # 获取数值特征
        this_num = self.X_num[index]

        # 获取分类特征（如果存在）
        if self.X_cat is not None:
            this_cat = self.X_cat[index]
        else:
            this_cat = []  # 无分类特征时返回空列表

        # 获取标签（如果存在）
        if self.y is not None:
            this_y = self.y[index]
            return (this_num, this_cat, this_y)

        return (this_num, this_cat)

    def __len__(self):
        """
        返回数据集大小

        返回:
            int: 样本总数
        """
        return self.X_num.shape[0]

def preprocess(dataset_path, task_type='binclass', inverse=False, cat_encoding=None, num_encoding='min_max_torch'):
    """
    数据预处理的主函数，负责加载、转换和返回训练/测试数据

    参数:
        dataset_path (str): 数据集目录路径，包含 .npy 文件和 info.json
        task_type (str): 任务类型，可选值:
            - 'binclass': 二分类
            - 'multiclass': 多分类
            - 'regression': 回归
        inverse (bool): 是否返回逆变换函数（用于将预测结果转换回原始尺度）
        cat_encoding (str, optional): 分类变量编码方式，可选值:
            - None: 不编码，保持原始分类特征
            - "one-hot": One-Hot编码
            - "one-hot-drop-bin": One-Hot编码并删除二值特征的一个维度
        num_encoding (str): 数值变量归一化方式，默认 'min_max_torch'

    返回:
        根据 cat_encoding 和 inverse 参数的不同，返回不同的元组:

        情况1: cat_encoding=None, inverse=False
            (X_num, X_cat, categories, d_numerical, y)

        情况2: cat_encoding=None, inverse=True
            (X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse, y)

        情况3: cat_encoding="one-hot", inverse=False
            (X_num, None, None, None, y)

        情况4: cat_encoding="one-hot", inverse=True
            (X_num, None, None, None, num_inverse, cat_inverse, y)

    示例:
        # 基本使用
        X_num, X_cat, categories, d_numerical, y = preprocess("data/adult")

        # 需要逆变换时
        X_num, X_cat, categories, d_numerical, num_inv, cat_inv, y = preprocess(
            "data/adult", inverse=True
        )
    """

    # ==================== 配置数据转换参数 ====================
    T_dict = {}

    # 数值归一化方式: min_max_torch, standard, robust, quantile_normal, etc.
    T_dict['normalization'] = num_encoding

    # 数值特征缺失值填充策略: 'mean', 'median', 'max', 'min', 'zero'
    T_dict['num_nan_policy'] = 'mean'

    # 分类特征缺失值填充策略: None表示不处理
    T_dict['cat_nan_policy'] = None

    # 分类特征最小频率阈值: 低于此频率的类别会被合并为"其他"
    T_dict['cat_min_frequency'] = None

    # 分类特征编码方式: None, "one-hot", "one-hot-drop-bin", "ordinal"
    T_dict['cat_encoding'] = cat_encoding

    # 标签处理策略: "default"表示按info.json中的配置处理
    T_dict['y_policy'] = "default"

    # 创建转换器对象
    T = src.Transformations(**T_dict)

    # ==================== 加载数据集 ====================
    dataset = make_dataset(
        data_path=dataset_path,
        T=T,
        change_val=False,  # 是否从训练集中划分验证集
    )

    # ==================== 处理不同编码情况 ====================

    # 情况1: 不对分类变量编码，保留原始分类特征
    if cat_encoding is None:
        # 从数据集对象中提取数据
        X_num = dataset.X_num  # dict: {'train': array, 'test': array}
        X_cat = dataset.X_cat  # dict: {'train': array, 'test': array}

        # 拆分训练集和测试集
        X_train_num, X_test_num = X_num['train'], X_num['test']
        X_train_cat, X_test_cat = X_cat['train'], X_cat['test']

        # 获取分类特征的类别信息（用于后续编码或逆变换）
        # categories: list of arrays，每个元素是对应分类特征的唯一值
        categories = src.get_categories(X_train_cat)

        # 获取数值特征的维度
        d_numerical = X_train_num.shape[1]

        # 打包为元组便于返回
        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)

        # 获取标签
        y = (dataset.y['train'], dataset.y['test'])

        # 如果需要逆变换函数
        if inverse:
            # 获取数值特征的逆变换函数（如果存在）
            if dataset.num_transform is not None:
                num_inverse = dataset.num_transform.inverse_transform
            else:
                num_inverse = None

            # 获取分类特征的逆变换函数
            cat_inverse = dataset.cat_transform.inverse_transform

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse, y
        else:
            return X_num, X_cat, categories, d_numerical, y

    # 情况2: One-Hot编码（编码后分类特征合并到数值特征中）
    elif cat_encoding == "one-hot" or cat_encoding == "one-hot-drop-bin":
        # One-Hot编码后，所有特征都在X_num中
        X_num = dataset.X_num

        # 拆分训练集和测试集
        X_train_num, X_test_num = X_num['train'], X_num['test']

        # 打包为元组
        X_num = (X_train_num, X_test_num)

        # 获取标签
        y = (dataset.y['train'], dataset.y['test'])

        # 如果需要逆变换函数
        if inverse:
            num_inverse = dataset.num_transform.inverse_transform
            cat_inverse = dataset.cat_transform.inverse_transform

            # One-Hot编码后，X_cat, categories, d_numerical 都为 None
            return X_num, None, None, None, num_inverse, cat_inverse, y
        else:
            return X_num, None, None, None, y

    # 情况3: 其他编码方式，直接返回原始数据集对象
    else:
        return dataset


def make_dataset(
    data_path: str,
    T: src.Transformations,
    change_val: bool,
):
    """
    从磁盘加载数据并构建数据集对象

    数据目录结构:
        data_path/
        ├── X_num_train.npy    # 训练集数值特征 [n_train, n_numerical]
        ├── X_num_test.npy     # 测试集数值特征 [n_test, n_numerical]
        ├── X_cat_train.npy    # 训练集分类特征 [n_train, n_categorical] (可选)
        ├── X_cat_test.npy     # 测试集分类特征 [n_test, n_categorical] (可选)
        ├── y_train.npy        # 训练集标签 [n_train, 1]
        ├── y_test.npy         # 测试集标签 [n_test, 1]
        └── info.json          # 数据集元信息

    info.json 示例:
        {
            "task_type": "binclass",
            "n_classes": 2,
            "target_class": "收入>50K",
            "negative_class": "收入<=50K",
            "num_features": ["年龄", "教育年限", "工作时长"],
            "cat_features": ["职业", "婚姻状况", "教育程度"]
        }

    参数:
        data_path (str): 数据集目录路径
        T (src.Transformations): 数据转换器对象，包含归一化、编码等配置
        change_val (bool): 是否从训练集中划分验证集

    返回:
        src.Dataset: 数据集对象，包含:
            - X_num: 数值特征字典 {'train': array, 'test': array}
            - X_cat: 分类特征字典 {'train': array, 'test': array} 或 None
            - y: 标签字典 {'train': array, 'test': array} 或 None
            - task_type: 任务类型
            - n_classes: 类别数
    """

    # ==================== 检查数据文件是否存在 ====================
    # 根据文件存在性初始化数据字典，不存在则为None
    X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
    X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
    y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

    # ==================== 读取训练和测试数据 ====================
    for split in ['train', 'test']:
        # 从npy文件读取原始数据
        X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)

        # 按照文件存在性存储数据
        if X_num is not None:
            X_num[split] = X_num_t
        if X_cat is not None:
            X_cat[split] = X_cat_t
        if y is not None:
            y[split] = y_t

    # ==================== 加载数据集元信息 ====================
    info = src.load_json(os.path.join(data_path, 'info.json'))

    # ==================== 创建数据集对象 ====================
    D = src.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},  # 标签的额外信息（如回归任务的目标范围）
        task_type=src.TaskType(info['task_type']),  # 转换为枚举类型
        n_classes=info.get('n_classes')  # 类别数（分类任务）
    )

    # ==================== 可选：划分验证集 ====================
    if change_val:
        D = src.change_val(D)

    # ==================== 应用数据转换并返回 ====================
    # 转换包括: 归一化、缺失值填充、分类变量编码等
    return src.transform_dataset(D, T, None)
