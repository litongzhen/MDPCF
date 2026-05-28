import numpy as np
import pandas as pd
import os
import sys
import json
import argparse
import pandas.api.types as ptypes

# 类型映射字典
TYPE_TRANSFORM = {"float", np.float32, "str", str, "int", int}

# 配置文件存放路径
INFO_PATH = "data/Info"

# 命令行参数解析器
parser = argparse.ArgumentParser(description="process dataset")
parser.add_argument("--dataname", type=str, default=None, help="Name of dataset.")
args = parser.parse_args()


# idx_mapping: original idx -> new idx 让数值列在前，类别列在后，最后是目标列
# inverse_idx_mapping: new idx -> original idx
# idx_name_mapping: original idx -> column name
def get_column_name_mapping(
    data_df, num_col_idx, cat_col_idx, target_col_idx, column_names=None
):

    # 如果没传列名，就直接从 DataFrame 拿
    if not column_names:
        column_names = np.array(data_df.columns.tolist())

    idx_mapping = {}  # 存放 {旧索引: 新索引}

    # 设定新索引的起始位置
    curr_num_idx = 0  # 数值列从 0 开始放
    curr_cat_idx = len(num_col_idx)  # 类别列紧接着数值列后面放
    curr_target_idx = curr_cat_idx + len(cat_col_idx)  # 目标列放在最后

    for idx in range(len(column_names)):  # 遍历原始数据的每一列（0, 1, 2, ...）

        if idx in num_col_idx:
            # 如果当前列是数值型
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1  # 数值计数器 +1
        elif idx in cat_col_idx:
            # 如果当前列是类别型
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1  # 类别计数器 +1
        else:
            # 或者是目标列
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1  # 目标计数器 +1

    # 反向映射： {新索引: 旧索引}
    # 以后如果想知道“矩阵里的第2列原来叫什么名字”，就用这个查
    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k

    # 名称映射： {旧索引: 列名字符串}
    # 方便打印调试，比如知道 0 对应 "age"
    idx_name_mapping = {}
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train=0, num_test=0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)  # 生成索引 [0, 1, 2, ... N]

    seed = 1234  # 初始随机种子

    while True:  # 死循环，直到满意为止
        np.random.seed(seed)
        np.random.shuffle(idx)  # 1. 随机打乱索引

        # 2. 根据数量切分
        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.iloc[train_idx]
        test_df = data_df.iloc[test_idx]

        flag = 0
        # 3. 核心检查逻辑
        for i in cat_columns:
            # 如果 训练集里的类别种类数 != 原始数据里的类别种类数
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1  # 标记为失败
                break

        # 4. 判定
        if flag == 0:
            break  # 成功！训练集涵盖了所有类别，跳出循环
        else:
            seed += 1  # 失败！换个随机种子，重头再来

    return train_df, test_df, seed


def process_data(name):
    # 读取对应的 json 配置文件
    with open(f"{INFO_PATH}/{name}.json", "r") as f:
        info = json.load(f)

    # 根据配置加载 CSV 或 Excel
    data_path = info["data_path"]
    if info["file_type"] == "csv":
        data_df = pd.read_csv(data_path, header=info["header"], low_memory=False)
    elif info["file_type"] == "xls":
        data_df = pd.read_excel(data_path, sheet_name="Data", header=1)
        data_df = data_df.drop("ID", axis=1)  # Excel通常有ID列，删掉

    # 清理 Pandas 保存时自动生成的 Unnamed 索引列
    unnamed_cols = [col for col in data_df.columns.map(str) if "Unnamed" in col]
    data_df = data_df.drop(columns=unnamed_cols)

    # 如果是 Lending Club 数据集
    if name == "lending-club":
        selected_columns = info["column_names"]
        data_df = data_df[selected_columns]  # 只保留需要的列
        data_df = data_df.dropna()  # 删除任何有空值的行（暴力清洗）

    # 预处理特定列的值（数据清洗）
    # 比如：只保留 "Married", "Single" 两种状态，其他的都改成 "Other"
    if "preprocess_columns" in info:
        for col, valid in info["preprocess_columns"].items():
            if name == "lending-club":
                # 对于 lending-club，不在有效列表里的变成 None
                data_df[col] = data_df[col].apply(lambda x: x if x in valid else None)
            else:
                # 对于其他数据，不在有效列表里的变成 'Other'
                data_df[info["column_names"].index(col)] = data_df[
                    info["column_names"].index(col)
                ].apply(lambda x: x if x in valid else "Other")

    # 再次清洗和截断数据
    if name == "lending-club":
        data_df = data_df.dropna()
        data_df = data_df[: info["samples_keep"]]  # 只取前 N 个样本（为了跑得快）

    target_col = info["target_col"]

    # 删除忽略的列（ignore_columns）
    ignore_columns = info["ignore_columns"]
    ignore_columns_index = [info["column_names"].index(x) for x in ignore_columns]
    data_df = data_df.drop(ignore_columns_index, axis=1)

    # 确保 Target 列在最后（或者单纯提取出来）
    if target_col in data_df:
        data_df = data_df[[col for col in data_df if col != target_col] + [target_col]]

    # GMC数据集特殊处理
    if name == "gmc":
        # GMC 数据集如果不清洗直接跑会报错，所以这里强制删除含空值的行
        data_df = data_df.dropna()
        # [cite_start]为了实验公平性和速度，只保留配置文件中规定的样本数量（如 30,000 条）[cite: 235]
        data_df = data_df[: info["samples_keep"]]

    # 更新数据行数，用于后续计算切分比例
    num_data = data_df.shape[0]

    # 确定最终使用的列名列表
    # 优先使用 json 配置文件里的 "column_names"，如果没有则使用 DataFrame 自带的列名
    column_names = (
        info["column_names"] if info["column_names"] else data_df.columns.tolist()
    )

    # 遍历列名，把出现在 "ignore_columns"（如 ID 列、无用特征）里的列剔除掉
    column_names = [x for x in column_names if x not in ignore_columns]
    # 识别每一列的数据类型
    # 如果 json 里已经写死了索引（num_col_idx 等），直接读取
    if "num_col_idx" in info:
        num_col_idx = info["num_col_idx"]  # 数值列的索引列表
        cat_col_idx = info["cat_col_idx"]  # 类别列的索引列表
        target_col_idx = info["target_col_idx"]  # 目标列（标签）的索引列表
    else:
        # 如果没写索引，就自动推断：
        # 遍历 column_names，去 json 的 "column_info" 字典里查，是 "float" 还是 "str"
        num_col_idx = [
            idx
            for idx, col in enumerate(column_names)
            if info["column_info"][col] == "float"
        ]
        # 类别列：标记为 "str" 且不是目标列的
        cat_col_idx = [
            idx
            for idx, col in enumerate(column_names)
            if info["column_info"][col] == "str" and col != target_col
        ]
        # 目标列：类别等于 target_col 的
        target_col_idx = [
            idx for idx, col in enumerate(column_names) if col == target_col
        ]

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
        data_df, num_col_idx, cat_col_idx, target_col_idx, column_names
    )
    # 根据索引提取出具体的列名列表
    num_columns = [column_names[i] for i in num_col_idx]  # 所有数值列名
    cat_columns = [column_names[i] for i in cat_col_idx]  # 所有类别列名
    target_columns = [column_names[i] for i in target_col_idx]  # 目标列名

    # 处理训练集和测试集
    if info["test_path"]:
        # --- 分支 A：如果有指定的测试文件（如 Adult 数据集） ---
        test_path = info["test_path"]

        # 读取测试文件，手动去掉文件末尾可能存在的句号（特定格式清洗）
        with open(test_path, "r") as f:
            lines = f.readlines()[1:]  # 跳过第一行（通常是 Header）
            test_save_path = f"data/{name}/test.data"
            if not os.path.exists(test_save_path):
                with open(test_save_path, "a") as f1:
                    for line in lines:
                        # 去掉换行符和末尾的 '.'
                        save_line = line.strip("\n").strip(".")
                        f1.write(f"{save_line}\n")

        # 读取处理好的测试数据，无表头
        test_df = pd.read_csv(test_save_path, header=None)
        # 同样要删掉忽略的列
        test_df = test_df.drop(ignore_columns_index, axis=1)

        # 对测试集进行特定的列预处理（同训练集逻辑）
        if "preprocess_columns" in info:
            for col, valid in info["preprocess_columns"].items():
                test_df[info["column_names"].index(col)] = test_df[
                    info["column_names"].index(col)
                ].apply(lambda x: x if x in valid else "Other")

        # 如果有单独的测试文件，那 data_df 全都是训练集
        train_df = data_df

    else:
        # --- 分支 B：如果没有测试文件（如 Lending Club） ---
        # 需要把 data_df 切分成训练集和测试集

        # 确定训练样本数量：优先读配置，没有则默认 90%
        if "train_test_split" in info:
            num_train = int(num_data * float(info["train_test_split"]))
        else:
            num_train = int(num_data * 0.9)
        num_test = num_data - num_train

        # 调用自定义切分函数：保证切分后的训练集包含所有类别特征的取值
        train_df, test_df, seed = train_val_test_split(
            data_df, cat_columns, num_train, num_test
        )

    # 重置 DataFrame 的列名为 0, 1, 2... 索引格式
    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))

    # 统计每一列的元数据
    col_domain = {}  # 存放每一列的值域（Domain）
    col_info = {}  # 存放每一列的详细类型信息

    # 统计数值列：记录最大值和最小值
    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "numerical"
        col_info["max"] = float(train_df[col_idx].max())  # 获取最大值
        col_info["min"] = float(train_df[col_idx].min())  # 获取最小值
        col_name = column_names[col_idx]
        col_domain[col_name] = [
            float(train_df[col_idx].min()),
            float(train_df[col_idx].max()),
        ]

    # 统计类别列：记录所有出现过的唯一值（Categorizes）
    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "categorical"
        col_info["categorizes"] = list(set(train_df[col_idx]))  # 获取去重后的类别列表
        col_name = column_names[col_idx]
        col_domain[col_name] = train_df[col_idx].unique().tolist()

    # 统计目标列：通常也是类别型（分类任务）
    for col_idx in target_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "categorical"
        col_info["categorizes"] = list(set(train_df[col_idx]))

    # 针对不同数据集，硬编码指定哪个类别是我们要生成的 Counterfactual 目标
    if name == "adult":
        info["target_class"] = " >50K"  # 目标：收入大于 50K
        info["negative_class"] = " <=50K"
    elif name == "lending-club":
        info["target_class"] = "Fully Paid"  # 目标：还清贷款
        info["negative_class"] = "Charged Off"
    elif name == "bank":
        info["target_class"] = "yes"  # 目标：订阅定期存款
        info["negative_class"] = "no"
    else:
        # 默认情况（如 Credit Default 或 GMC）
        print("Warning: Setting target class for dataset to 1")
        info["target_class"] = 1
        info["negative_class"] = 0

    # 将统计好的信息存回 info 字典
    info["column_info"] = col_info
    info["column_domain"] = col_domain

    # 将 DataFrame 的数字索引列名重命名回真实的字符串列名
    # 这是为了下面处理 NaN 值时可以通过列名来操作
    train_df.rename(columns=idx_name_mapping, inplace=True)
    test_df.rename(columns=idx_name_mapping, inplace=True)

    for df in [train_df, test_df]:
        # 把所有【数值列】里的 '?' 替换为 np.nan
        df[num_columns] = df[num_columns].replace("?", np.nan)
        # 把所有【类别列】里的 '?' 替换为字符串 'nan'
        df[cat_columns] = df[cat_columns].replace("?", "nan")

    # DataFrame -> Numpy Array
    # 数值特征转 float32 (神经网络通用格式)
    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    # 类别特征保持原始格式 (后续会做 Tokenization/Embedding)
    X_cat_train = train_df[cat_columns].to_numpy()
    # 标签
    y_train = train_df[target_columns].to_numpy()

    # 记录输入列的物理顺序（用于后续还原）
    input_col_order = num_columns + cat_columns

    # (测试集同理)
    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()

    # 保存文件
    save_dir = f"data/{name}"

    # 保存 Numpy 矩阵 (.npy) - 这是模型训练直接读取的文件
    np.save(f"{save_dir}/X_num_train.npy", X_num_train)
    np.save(f"{save_dir}/X_cat_train.npy", X_cat_train)
    np.save(f"{save_dir}/y_train.npy", y_train)

    np.save(f"{save_dir}/X_num_test.npy", X_num_test)
    np.save(f"{save_dir}/X_cat_test.npy", X_cat_test)
    np.save(f"{save_dir}/y_test.npy", y_test)

    # 确保数值列的数据类型是 float32 (为了保险起见，再次转换)
    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)

    # 同时也保存一份处理好的 CSV 文件，方便查看
    train_df.to_csv(f"{save_dir}/train.csv", index=False)
    test_df.to_csv(f"{save_dir}/test.csv", index=False)

    # if not os.path.exists(f'synthetic/{name}'):
    #     os.makedirs(f'synthetic/{name}')

    # train_df.to_csv(f'synthetic/{name}/real.csv', index = False)
    # test_df.to_csv(f'synthetic/{name}/test.csv', index = False)

    # 打印一些形状信息，确认没出错
    print("Numerical", X_num_train.shape)
    print("Categorical", X_cat_train.shape)

    # 更新 info 字典，加入最大最小值（字符串格式，为了写入 JSON）
    info["num_max"] = dict(zip(num_columns, X_num_train.max(axis=0).astype(str)))
    info["num_min"] = dict(zip(num_columns, X_num_train.min(axis=0).astype(str)))

    # 记录最终的数据集规模和映射关系
    info["column_names"] = column_names
    info["train_num"] = train_df.shape[0]
    info["test_num"] = test_df.shape[0]
    info["input_col_order"] = input_col_order
    info["idx_mapping"] = idx_mapping
    info["inverse_idx_mapping"] = inverse_idx_mapping
    info["idx_name_mapping"] = idx_name_mapping

    # 生成详细的 Metadata 结构（符合 SDV 等库的标准格式）
    metadata = {"columns": {}}
    task_type = info["task_type"]

    info["num_col_idx"] = num_col_idx
    info["cat_col_idx"] = cat_col_idx
    info["target_col_idx"] = target_col_idx

    # num_col_idx = info['num_col_idx']
    # cat_col_idx = info['cat_col_idx']
    # target_col_idx = info['target_col_idx']

    # 循环写入每一列的 sdtype: numerical 或 categorical
    for i in num_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "numerical"
        metadata["columns"][i]["computer_representation"] = "Float"

    for i in cat_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "categorical"

    for i in target_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "categorical"

    info["metadata"] = metadata

    # 最终保存 info.json 配置文件
    with open(f"{save_dir}/info.json", "w") as file:
        json.dump(info, file, indent=4)

    print(f"Processing and Saving {name} Successfully!")

    print(name)
    print("Total", info["train_num"] + info["test_num"])
    print("Train", info["train_num"])
    print("Test", info["test_num"])

    cat = len(cat_col_idx)
    num = len(num_col_idx)
    print("Num", num)
    print("Cat", cat)

    print("Numeric", num_columns)
    print("Categorical", cat_columns)
    print("Classification Target", target_columns)


if __name__ == "__main__":

    if args.dataname:
        process_data(args.dataname)
    else:
        for name in [
            "adult"
        ]:  # , 'adult', 'default', 'lending-club', 'gmc']:#, 'default', 'shoppers', 'magic']:
            process_data(name)
