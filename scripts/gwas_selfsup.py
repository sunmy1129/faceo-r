#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自监督图表示学习模块（v2.0 核心修复）
- 任务 A：边预测（遮蔽 15% 边 + 等量负采样，BCE 损失）
- 不使用任何社区标签，因此后续「社区在嵌入空间中是否分离」构成独立检验
- 输出：嵌入矩阵（默认 64 维）
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import negative_sampling, dropout_edge
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

SEED = 42


class GATEncoder(nn.Module):
    """带 edge_dim 的 GAT 编码器 —— 修复 v1.0 中 edge_attr 被忽略的 bug"""

    def __init__(self, in_ch, hidden=64, out=64, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_ch, hidden, heads=heads, edge_dim=1, dropout=0.2)
        self.conv2 = GATConv(hidden * heads, out, heads=1, edge_dim=1, dropout=0.2)

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x


class SAGEEncoder(nn.Module):
    """GraphSAGE 对照编码器"""

    def __init__(self, in_ch, hidden=64, out=64):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden)
        self.conv2 = SAGEConv(hidden, out)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def build_pyg_data(tom, node_feat=None, edge_percentile=20):
    """TOM -> PyG Data（稀疏化：保留前 (100-percentile)% 的边）"""
    n = tom.shape[0]
    thr = np.percentile(tom[tom > 0], edge_percentile)
    rows, cols = np.where(tom > thr)
    mask = rows < cols
    rows, cols = rows[mask], cols[mask]
    w = tom[rows, cols]
    ei = torch.from_numpy(np.vstack([rows, cols])).long()
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    ea = torch.from_numpy(np.concatenate([w, w])).float().unsqueeze(1)
    if node_feat is None:
        node_feat = tom.copy()
    x = torch.from_numpy(np.asarray(node_feat, dtype=np.float32))
    # 标准化
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-8)
    return x, ei, ea


def train_edge_prediction(x, ei, ea, encoder="gat", epochs=300, lr=0.01,
                          hidden=64, out_dim=64, mask_frac=0.15, seed=SEED,
                          patience=40, verbose=False):
    """
    自监督边预测：遮蔽部分边训练，评估用全部边
    返回: embedding (numpy), 训练损失曲线
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = x.shape[0]
    if encoder == "gat":
        model = GATEncoder(x.shape[1], hidden, out_dim)
    else:
        model = SAGEEncoder(x.shape[1], hidden, out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # 手动边遮蔽（避免 dropout_edge 的掩码形状问题）
    n_edges = ei.shape[1]
    perm = torch.randperm(n_edges)
    n_mask = max(1, int(n_edges * mask_frac))
    mask_idx, keep_idx = perm[:n_mask], perm[n_mask:]
    ei_keep = ei[:, keep_idx]
    ea_keep = ea[keep_idx]
    pos_edges = ei[:, mask_idx].unique(dim=1)  # 无向去重
    losses, best_loss, best_state, wait = [], np.inf, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        z = model(x, ei_keep, ea_keep)
        neg = negative_sampling(edge_index=ei_keep, num_nodes=n,
                                num_neg_samples=pos_edges.shape[1])
        pos_score = (z[pos_edges[0]] * z[pos_edges[1]]).sum(-1)
        neg_score = (z[neg[0]] * z[neg[1]]).sum(-1)
        loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()
        loss.backward()
        opt.step()
        l = float(loss.detach())
        losses.append(l)
        if l < best_loss - 1e-5:
            best_loss, wait = l, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb = model(x, ei, ea).cpu().numpy()
    return emb, losses


def embed_to_partition(emb, n_clusters=None, seed=SEED):
    """嵌入 -> 划分（最近质心法 / KMeans）"""
    k = n_clusters or max(2, min(8, emb.shape[0] // 20))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(emb)


def consistency_with_partition(emb, labels, seed=SEED):
    """
    独立验证：嵌入的聚类结果与给定划分（如 Leiden 社区）的 ARI
    由于嵌入训练不使用这些标签，该 ARI 构成独立证据
    """
    labs = np.asarray(labels)
    k = len(set(labs[labs >= 0]))
    if k < 2 or k >= emb.shape[0]:
        return np.nan
    pred = embed_to_partition(emb, k, seed)
    return float(adjusted_rand_score(labs, pred))


if __name__ == "__main__":
    # 自检：构造已知 3 社区图，验证自监督嵌入能恢复结构
    rng = np.random.default_rng(0)
    n_per, k = 30, 3
    n = n_per * k
    true_lab = np.repeat(np.arange(k), n_per)
    # 块状 TOM
    tom = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.6 if true_lab[i] == true_lab[j] else 0.05
            v = rng.random()
            tom[i, j] = tom[j, i] = (0.5 + 0.5 * rng.random()) if v < p else 0.0
    x, ei, ea = build_pyg_data(tom)
    emb, losses = train_edge_prediction(x, ei, ea, epochs=300, out_dim=32)
    print(f"训练损失: {losses[0]:.3f} -> {losses[-1]:.3f} (共 {len(losses)} 轮)")
    ari = consistency_with_partition(emb, true_lab)
    print(f"嵌入聚类 vs 真实社区 ARI = {ari:.3f}  (自检通过标准 >0.6)")
    if ari > 0.6:
        print("✅ 自监督嵌入管线正常，可用于真实数据")
    else:
        print("⚠️ 嵌入质量不足，需调整超参数")
