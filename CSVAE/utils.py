import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 设置中文字体 ---
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


def generate_equiangular_unit_vectors(n, d=None):
    """
    在 d 维单位球面上生成 n 个点，使得任意两点之间的夹角相等。
    """
    if d is None:
        d = n - 1
    if d < n - 1:
        raise ValueError(f"需要 d >= n-1（当前 n={n}, d={d}）")

    # Step 1: 构造 Gram 矩阵
    G = np.full((n, n), -1.0 / (n - 1))
    np.fill_diagonal(G, 1.0)

    # Step 2: 特征分解
    eigenvals, eigenvecs = np.linalg.eigh(G)

    # Step 3: 提取特征向量并归一化
    X = eigenvecs[:, 1 : 1 + d] * np.sqrt(eigenvals[1])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = X / norms

    return X


def plot_feasibility_experiment_fixed():
    """
    绘制单纯形采样的可行性验证图 (修正球体比例版)
    """
    fig = plt.figure(figsize=(15, 7))
    plt.suptitle("基于单纯形采样的多样性验证", fontsize=18, fontweight="bold")

    # --- 实验 1: 2D 平面 (N=3) ---
    ax1 = fig.add_subplot(121)
    N1 = 3
    X1 = generate_equiangular_unit_vectors(N1, d=2)

    # 计算实际余弦值
    dots1 = X1 @ X1.T
    mask1 = ~np.eye(N1, dtype=bool)
    avg_cos1 = np.mean(dots1[mask1])
    theory_cos1 = -1.0 / (N1 - 1)

    # 绘制
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(
        np.cos(theta),
        np.sin(theta),
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="单位圆",
    )

    colors = ["#E63946", "#457B9D", "#1D3557"]
    for i in range(N1):
        ax1.quiver(
            0,
            0,
            X1[i, 0],
            X1[i, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=colors[i],
            width=0.012,
            label=f"向量 $z_{i+1}$",
        )
        ax1.text(
            X1[i, 0] * 1.15,
            X1[i, 1] * 1.15,
            f"$z_{i+1}$",
            fontsize=14,
            fontweight="bold",
            color=colors[i],
            ha="center",
        )

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect("equal")  # 2D 图已经设置了 equal，所以圆是圆的
    ax1.grid(True, linestyle=":", alpha=0.6)

    title_text_1 = (
        f"案例 1: 2D 潜空间 ($N=3$)\n"
        f"实际余弦值: {avg_cos1:.4f} (理论值: {theory_cos1:.4f})\n"
        f"几何结构: 等边三角形 (最大化互斥)"
    )
    ax1.set_title(title_text_1, fontsize=12)
    ax1.legend(loc="lower right", fontsize=10)

    # --- 实验 2: 3D 空间 (N=4) ---
    ax2 = fig.add_subplot(122, projection="3d")
    N2 = 4
    X2 = generate_equiangular_unit_vectors(N2, d=3)

    dots2 = X2 @ X2.T
    mask2 = ~np.eye(N2, dtype=bool)
    avg_cos2 = np.mean(dots2[mask2])
    theory_cos2 = -1.0 / (N2 - 1)

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_wireframe(x, y, z, color="gray", alpha=0.1)

    colors_3d = ["#E63946", "#457B9D", "#1D3557", "#A8DADC"]
    for i in range(N2):
        ax2.plot(
            [0, X2[i, 0]], [0, X2[i, 1]], [0, X2[i, 2]], color=colors_3d[i], linewidth=2
        )
        ax2.scatter(
            X2[i, 0], X2[i, 1], X2[i, 2], color=colors_3d[i], s=80, label=f"$z_{i+1}$"
        )
        ax2.text(
            X2[i, 0] * 1.1,
            X2[i, 1] * 1.1,
            X2[i, 2] * 1.1,
            f"$z_{i+1}$",
            fontsize=12,
            fontweight="bold",
            color=colors_3d[i],
        )

    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)

    # ==========================================
    # 关键修改：强制 3D 轴比例为 1:1:1
    # ==========================================
    ax2.set_box_aspect([1, 1, 1])

    title_text_2 = (
        f"案例 2: 3D 潜空间 ($N=4$)\n"
        f"实际余弦值: {avg_cos2:.4f} (理论值: {theory_cos2:.4f})\n"
        f"几何结构: 正四面体 (最大化互斥)"
    )
    ax2.set_title(title_text_2, fontsize=12)

    plt.tight_layout()
    plt.savefig("simplex_experiment_fixed.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_feasibility_experiment_fixed()
