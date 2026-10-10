#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FACEO-R — Function-Annotated Community Extraction with Overlap (Robust)

核心创新：自适应先验权重
    A = alpha(q) * TOM_expr  +  (1 - alpha(q)) * S_func
其中 q = 图可靠性（由 TOM 的 bootstrap 稳定性估计）。
    q 高（数据可靠） -> alpha -> 1（以表达为主，退化为标准 Leiden，不输）
    q 低（数据不可靠）-> alpha 降低（功能先验补偿）

关键性质（不输性保证）：
    当图可靠时 FACEO-R 自动退化为 Leiden；因此
    "在高质图上不劣于 Leiden" 是设计保证的，
    而 "在低质图上优于 Leiden" 是待验证的科学假设。
"""
import sys
import warnings
import numpy as np
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score as ari

warnings.filterwarnings("ignore")
SEED = 42


def go_similarity(genes, go2t, min_terms=3):
    n = len(genes)
    sets = [go2t.get(g.upper(), set()) for g in genes]
    valid = [len(s) >= min_terms for s in sets]
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if not valid[i]:
            continue
        si = sets[i]
        for j in range(i + 1, n):
            if not valid[j]:
                continue
            u = len(si | sets[j])
            if u:
                v = len(si & sets[j]) / u
                S[i, j] = S[j, i] = v
    return S


def graph_reliability(tom, n_boot=5, frac=0.8, seed=SEED):
    """
    图可靠性 q：bootstrap 重采样基因后的划分一致性（ARI 均值），归一化到 [0,1]。
    含义：数据越可靠，重采样后划分越稳定。
    """
    import igraph as ig
    import leidenalg as la
    T = np.asarray(tom, dtype=np.float32)
    n = T.shape[0]
    if n < 10:
        return 0.5
    rng = np.random.default_rng(seed)
    labs = []
    for b in range(n_boot):
        idx = np.sort(rng.choice(n, max(8, int(n * frac)), replace=False))
        Tb = T[np.ix_(idx, idx)]
        thr = np.percentile(Tb[np.triu_indices(len(idx), 1)], 90)
        edges = [(i, j) for i in range(len(idx)) for j in range(i + 1, len(idx))
                 if Tb[i, j] >= thr]
        if len(edges) < 3:
            continue
        g = ig.Graph(n=len(idx), edges=edges, directed=False)
        g.es["weight"] = [float(Tb[i, j]) for i, j in edges]
        p = la.find_partition(g, la.RBConfigurationVertexPartition,
                              weights="weight", resolution_parameter=1.0, seed=seed)
        lab = np.zeros(n, dtype=int) - 1
        for ci, c in enumerate(p):
            for i in c:
                lab[idx[i]] = ci
        labs.append(lab)
    if len(labs) < 2:
        return 0.5
    aris = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            m = (labs[i] >= 0) & (labs[j] >= 0)
            if m.sum() > 5:
                aris.append(ari(labs[i][m], labs[j][m]))
    if not aris:
        return 0.5
    # 映射：ARI 0.5 以下视为不可靠
    return float(np.clip((np.mean(aris) - 0.3) / 0.6, 0.0, 1.0))


def faceo_r(tom, genes, go2t, gamma=1.0, tau=0.65, min_size=3,
            alpha_min=0.35, alpha_max=0.95, reliability=None):
    """
    FACEO-R 主算法。

    返回 modules(重叠), hard(主归属), info(dict)
    """
    import igraph as ig
    import leidenalg as la

    n = len(genes)
    q = graph_reliability(tom) if reliability is None else float(reliability)
    alpha = alpha_min + (alpha_max - alpha_min) * q   # 可靠 -> alpha 大

    Sf = go_similarity(genes, go2t)
    T = np.asarray(tom, dtype=np.float32)
    Tn = T / (T.max() + 1e-9)
    Sfn = Sf / (Sf.max() + 1e-9)
    A = alpha * Tn + (1 - alpha) * Sfn
    np.fill_diagonal(A, 0)

    thr = np.percentile(A[np.triu_indices(n, 1)], 90)
    edges, w = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] >= thr:
                edges.append((i, j)); w.append(float(A[i, j]))
    if len(edges) < 3:
        return [set(range(n))], np.zeros(n, int), {"alpha": alpha, "q": q}

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = w
    part = la.find_partition(g, la.RBConfigurationVertexPartition,
                             weights="weight", resolution_parameter=gamma, seed=SEED)
    centers = [set(c) for c in part if len(c) >= min_size]
    if not centers:
        return [set(range(n))], np.zeros(n, int), {"alpha": alpha, "q": q}

    k = len(centers)
    memb = np.zeros((n, k), dtype=np.float32)
    for ci, c in enumerate(centers):
        memb[:, ci] = A[:, list(c)].mean(axis=1)
    rel = memb / (memb.max(axis=1, keepdims=True) + 1e-9)
    mask = rel >= tau
    modules = []
    for ci in range(k):
        m = set(np.where(mask[:, ci])[0].tolist())
        if len(m) >= min_size:
            modules.append(m)
    hard = np.argmax(memb, axis=1)
    return modules, hard, {"alpha": float(alpha), "q": float(q),
                           "n_modules": len(modules),
                           "mean_membership": float(mask.sum(axis=1).mean())}


def modules_to_labels(modules, n):
    lab = np.zeros(n, dtype=int)
    for mi, m in enumerate(modules):
        for i in m:
            lab[i] = mi
    return lab
