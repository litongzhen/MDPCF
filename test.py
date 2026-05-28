import sys
import numpy as np

class Node:
    def __init__(self, l, r, f, th, label):
        self.l = l
        self.r = r
        self.f = f 
        self.th = th 
        self.label = label

class Score:
    def __init__(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def __add__(self, other):
        score = Score()
        score.tp = self.tp + other.tp
        score.fp = self.fp + other.fp
        score.fn = self.fn + other.fn
        return score
    def f1(self):
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn) if 2 * self.tp + self.fp + self.fn > 0 else 0

def dfs(node: Node, data: np.ndarray) -> Score:
    if node.l == 0 and node.r == 0:
        score = Score()
        real = data[:, -1]
        if node.label == 1:
            score.tp = float(np.sum(real == 1))
            score.fp = float(np.sum(real == 0))
            score.fn = 0.0
        else:
            score.tp = 0.0
            score.fp = 0.0
            score.fn = float(np.sum(real == 1))
        return score
    data_left = data[data[node.f] <= node.th]
    data_right = data[data[node.f] > node.th]
    left, right = dfs(tree[Node.left], data_left), dfs(tree[Node.right], data_right)
    score = left + right
    score_stop_now = Score()
    real = data[:, -1]
    if node.label == 1:
        score_stop_now.tp = float(np.sum(real == 1))
        score_stop_now.fp = float(np.sum(real == 0))
        score_stop_now.fn = 0.0
    else:
        score_stop_now.tp = 0.0
        score_stop_now.fp = 0.0
        score_stop_now.fn = float(np.sum(real == 1))
    if score_stop_now.f1() > score.f1():
        return score_stop_now
    return score

def main():
    n, m, k = map(int, input().split())

    # tree[i] = [l, r, f, th, label]
    tree = [[]]
    for i in range(1, n + 1):
        l, r, f, th, label = list(map(int, input().split()))
        tree.append(Node(l, r, f, th, label))

    # 验证集，最后一列是真实标签
    valid_data = np.array([[] + list(map(int, input().split())) for _ in range(m)], dtype=np.int64)
    score = dfs(tree[1], valid_data)
    print(score.f1())


if __name__ == "__main__":
    main()