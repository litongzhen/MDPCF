import numpy as np

def generate_equiangular_unit_vectors(n, d=None):
    """
    在 d 维单位球面上生成 n 个点，使得任意两点之间的夹角相等（即正单纯形结构）。
    
    参数:
    n: 生成向量的数量
    d: 维数。如果不指定，默认为 n-1
    
    返回:
    X: 形状为 (n, d) 的阵列，每一行是一个单位向量
    """
    if d is None:
        d = n - 1
    if d < n - 1:
        raise ValueError(f"需要 d >= n-1（当前 n={n}, d={d}）才能保证等夹角性")

    # Step 1: 构造理论 Gram 矩阵 G
    # 对于 n 个等角向量，其两两之间的余弦值应为 -1/(n-1)
    G = np.full((n, n), -1.0 / (n - 1))
    np.fill_diagonal(G, 1.0)

    # Step 2: 对 Gram 矩阵进行特征分解
    # 因为 G 是对称矩阵，使用 eigh 更稳定
    eigenvals, eigenvecs = np.linalg.eigh(G)

    # Step 3: 提取特征向量并映射到目标维度
    # 我们选择最大的 d 个特征值对应的部分进行构造
    # 注意：由于 G 的秩为 n-1，最小的一个特征值理论上应为 0
    X = eigenvecs[:, 1 : 1 + d] * np.sqrt(np.maximum(eigenvals[1:], 0))

    # Step 4: 归一化处理（确保数值精度导致的微小偏差不影响单位长度）
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = X / norms

    return X

def verify_vectors(X):
    """
    验证生成的向量是否满足等夹角特性
    """
    n = X.shape[0]
    # 计算所有向量两两之间的内积（即余弦值）
    dot_products = X @ X.T
    
    # 提取非对角线元素
    mask = ~np.eye(n, dtype=bool)
    off_diag = dot_products[mask]
    
    avg_cos = np.mean(off_diag)
    theory_cos = -1.0 / (n - 1)
    
    print(f"向量数量: {n}, 维度: {X.shape[1]}")
    print(f"实际平均余弦值: {avg_cos:.6f}")
    print(f"理论预期余弦值: {theory_cos:.6f}")
    print(f"最大误差: {np.max(np.abs(off_diag - theory_cos)):.6e}")

if __name__ == "__main__":
    # 示例 1: 在 2D 空间生成 3 个点（等边三角形）
    vectors_2d = generate_equiangular_unit_vectors(3, d=2)
    verify_vectors(vectors_2d)

    print("-" * 30)

    # 示例 2: 在 10D 空间生成 11 个点
    vectors_10d = generate_equiangular_unit_vectors(11, d=10)
    verify_vectors(vectors_10d)