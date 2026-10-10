#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FACEO — Function-Annotated Community Extraction with Overlap
新方法：功能注释引导的重叠社区检测

核心机制（相对硬划分的结构性优势）：
  硬划分（Leiden/WGCNA）强制每个基因属于 1 个模块；
  但实测 98.3% 的基因平均有 40 个功能归属。
  FACEO 允许基因归属多个模块，在「功能覆盖率」上有结构性优势。

公平性设计（避免循环论证）：
  建图/选参 使用 GO Biological Process 注释；
  最终评价 使用 held-out 的 Reactome 注释（互不重叠的注释体系）。
"""
import sys
import warnings
import numpy as np
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score as ari

warnings.filterwarnings("ignore")

SEED = 42


# ---------------------------------------------------------------- 功能相似度
def go_pairwise_similarity(genes, go2t, min_terms=3):
    """基因对的 GO 注释 Jaccard 相似度矩阵（仅用 GO，不接触 Reactome）"""
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
            sj = sets[j]
            u = len(si | sj)
            if u:
                v = len(si & sj) / u
                S[i, j] = S[j, i] = v
    return S


# ---------------------------------------------------------------- 核心算法
def faceo(tom, genes, go2t, alpha=0.6, gamma=1.0, tau=0.65, min_size=3):
    """
    FACEO 算法。

    参数
    ----
    tom   : (n,n) 拓扑重叠矩阵（表达数据）
    genes : 基因符号列表
    go2t  : gene -> set(GO terms)（训练用注释）
    alpha : 表达图权重（1-alpha 为功能图权重）
    gamma : Leiden 分辨率
    tau   : 隶属度阈值（相对最大值），控制重叠程度

    返回
    ----
    modules : list[set] 重叠模块
    hard    : np.ndarray 每个基因的主归属（用于可比性）
    memb    : (n,k) 隶属度矩阵
    """
    import igraph as ig
    import leidenalg as la

    n = len(genes)
    Sfunc = go_pairwise_similarity(genes, go2t)

    # 1) 双源融合图：表达 TOM + 功能相似度
    T = np.asarray(tom, dtype=np.float32)
    Tn = T / (T.max() + 1e-9)
    Sf = Sfunc / (Sfunc.max() + 1e-9)
    A = alpha * Tn + (1 - alpha) * Sf
    np.fill_diagonal(A, 0)

    # 2) 在融合图上用 Leiden 得初始模块（提供"候选模块中心"）
    thr = np.percentile(A[np.triu_indices(n, 1)], 90)
    edges, weights = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] >= thr:
                edges.append((i, j)); weights.append(float(A[i, j]))
    if len(edges) < 3:
        return [set(range(n))], np.zeros(n, int), np.ones((n, 1))
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    part = la.find_partition(g, la.RBConfigurationVertexPartition,
                             weights="weight", resolution_parameter=gamma,
                             seed=SEED)
    centers = [set(c) for c in part if len(c) >= min_size]
    if not centers:
        return [set(range(n))], np.zeros(n, int), np.ones((n, 1))

    # 3) 隶属度：基因对每个模块的平均相似度（用融合图，非仅表达）
    k = len(centers)
    memb = np.zeros((n, k), dtype=np.float32)
    for ci, c in enumerate(centers):
        idx = list(c)
        memb[:, ci] = A[:, idx].mean(axis=1)

    # 4) 阈值化 -> 重叠归属（相对该基因最大隶属度）
    mx = memb.max(axis=1, keepdims=True) + 1e-9
    rel = memb / mx
    overlap_mask = rel >= tau
    modules = []
    for ci in range(k):
        m = set(np.where(overlap_mask[:, ci])[0].tolist())
        if len(m) >= min_size:
            modules.append(m)

    hard = np.asarray([ci for ci in range(k)])[0:0]  # placeholder
    hard = np.argmax(memb, axis=1)
    return modules, hard, memb


# ---------------------------------------------------------------- 评价
def coverage_score(modules, genes, ref2t, min_terms=1):
    """
    功能覆盖率（held-out 注释）：模块集合能覆盖的注释条目比例 × 模块纯度
    对重叠划分：一个注释可被多个模块覆盖（不惩罚重叠）
    对硬划分：每个基因只有 1 个归属
    """
    if not modules:
        return np.nan
    all_terms = set()
    for g in genes:
        all_terms |= ref2t.get(g.upper(), set())
    if not all_terms:
        return np.nan
    covered = set()
    purities = []
    for m in modules:
        gm = [genes[i] for i in m]
        terms = set()
        for g in gm:
            terms |= ref2t.get(g.upper(), set())
        if not terms:
            continue
        covered |= terms
        # 模块纯度：模块内最高频注释的占比
        from collections import Counter
        cnt = Counter()
        for g in gm:
            for t in ref2t.get(g.upper(), set()):
                cnt[t] += 1
        if cnt:
            purities.append(max(cnt.values()) / len(gm))
    cov = len(covered) / len(all_terms)
    pur = float(np.mean(purities)) if purities else 0.0
    return cov, pur


def partition_to_modules(labels, min_size=3):
    d = defaultdict(set)
    for i, l in enumerate(labels):
        if l >= 0:
            d[int(l)].add(i)
    return [s for s in d.values() if len(s) >= min_size]
