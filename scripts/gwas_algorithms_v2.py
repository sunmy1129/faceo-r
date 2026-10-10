#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法扩展模块：在 17 种基础算法之上新增 15 种
类别：多尺度稳定性 / 密度聚类 / 矩阵分解 / 重叠社区 / igraph 内置社区检测
"""
import warnings
import numpy as np
import igraph as ig
import networkx as nx
from sklearn.cluster import (DBSCAN, OPTICS, AffinityPropagation, MeanShift)
from sklearn.decomposition import NMF
import hdbscan

warnings.filterwarnings("ignore")
SEED = 42


def _tom_arr(g):
    return np.array(g.get_adjacency(attribute="weight").data, dtype=float)


def _distance(tom):
    """相似度 -> 距离矩阵（非负）"""
    d = 1.0 - tom
    np.fill_diagonal(d, 0)
    return np.clip(d, 0, 1)


# ---------- 多尺度稳定性 ----------
def alg_markov_stability(g, n_scales=25, seed=SEED):
    """Markov Stability（PyGenStability）：多尺度扫描，取最优尺度划分"""
    import pygenstability as pgs
    tom = _tom_arr(g)
    adj = (tom + tom.T) / 2
    np.fill_diagonal(adj, 0)
    adj = np.clip(adj, 0, None)
    if adj.sum() <= 0:
        return np.zeros(g.vcount(), dtype=int)
    try:
        res = pgs.run(adj, min_scale=-1.5, max_scale=0.5, n_scale=n_scales,
                      constructor="continuous_combinatorial", n_tries=30,
                      tqdm_disable=True, n_workers=1,
                      with_optimal_scales=True, with_postprocessing=False)
        cid = res.get("community_id")
        if cid is None or len(cid) == 0:
            return np.zeros(g.vcount(), dtype=int)
        # 选择模块数落在合理区间且 NVI 较低的尺度
        best_idx, best_score = 0, -1e9
        nvi = res.get("NVI")
        for i, lab in enumerate(cid):
            k = len(set(np.asarray(lab)))
            if k < 2 or k > max(20, g.vcount() // 5):
                continue
            stab = -float(nvi[i]) if nvi is not None and i < len(nvi) and nvi[i] == nvi[i] else 0.0
            # 偏好模块数 3-12
            pref = -abs(k - 6) * 0.1
            sc = stab + pref
            if sc > best_score:
                best_score, best_idx = sc, i
        return np.asarray(cid[best_idx])
    except Exception:
        return np.zeros(g.vcount(), dtype=int)


# ---------- 密度聚类 ----------
def _feat(tom):
    """基因特征空间：TOM 行向量（标准化）"""
    X = tom.copy()
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)
    return X


def _spectral_feat(tom, n_comp=8):
    """谱嵌入特征空间（降维后更适配密度聚类）"""
    from sklearn.manifold import SpectralEmbedding
    n = tom.shape[0]
    k = int(min(n_comp, max(2, n // 6), n - 2))
    if k < 2:
        return _feat(tom)
    try:
        se = SpectralEmbedding(n_components=k, affinity="precomputed",
                               random_state=SEED)
        return se.fit_transform(tom)
    except Exception:
        return _feat(tom)


def alg_dbscan(g, min_samples=3, seed=SEED):
    """DBSCAN：谱嵌入空间 + 自动 eps（k-距离拐点）"""
    from sklearn.neighbors import NearestNeighbors
    X = _spectral_feat(_tom_arr(g))
    n = X.shape[0]
    if n < min_samples + 1:
        return np.zeros(n, dtype=int)
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dists, _ = nn.kneighbors(X)
    kd = np.sort(dists[:, -1])
    # 网格搜索：选择使聚类数落在合理区间且轮廓系数最高的 eps
    from sklearn.metrics import silhouette_score
    best, best_sc = None, -1e9
    for q in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        eps = float(np.quantile(kd, q))
        lab = DBSCAN(eps=max(eps, 1e-4), min_samples=min_samples).fit_predict(X)
        k = len(set(lab[lab >= 0]))
        if k < 2 or k > max(15, X.shape[0] // 5):
            continue
        try:
            sc = silhouette_score(X, lab)
        except Exception:
            continue
        if sc > best_sc:
            best_sc, best = sc, lab
    return best if best is not None else DBSCAN(
        eps=max(float(np.median(kd)), 1e-3), min_samples=min_samples).fit_predict(X)


def alg_hdbscan(g, min_cluster_size=5):
    """HDBSCAN：谱嵌入空间"""
    X = _spectral_feat(_tom_arr(g))
    mcs = max(3, min(min_cluster_size, X.shape[0] // 8))
    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=2).fit_predict(X)


def alg_optics(g, min_samples=3):
    """OPTICS：谱嵌入空间 + 参数搜索"""
    from sklearn.metrics import silhouette_score
    X = _spectral_feat(_tom_arr(g))
    best, best_sc = None, -1e9
    for xi in (0.03, 0.05, 0.1, 0.2):
        for mcs in (3, 5, 8):
            try:
                lab = OPTICS(min_samples=min_samples, xi=xi,
                             min_cluster_size=mcs).fit_predict(X)
            except Exception:
                continue
            k = len(set(lab[lab >= 0]))
            if k < 2 or k > max(15, X.shape[0] // 5):
                continue
            try:
                sc = silhouette_score(X, lab)
            except Exception:
                continue
            if sc > best_sc:
                best_sc, best = sc, lab
    return best if best is not None else np.zeros(X.shape[0], dtype=int)


def alg_affinity(g, seed=SEED):
    """Affinity Propagation：以 TOM 直接作为相似度矩阵"""
    tom = _tom_arr(g)
    np.fill_diagonal(tom, np.median(tom[tom > 0]) if (tom > 0).any() else 0.5)
    try:
        return AffinityPropagation(affinity="precomputed", random_state=seed,
                                   max_iter=300, damping=0.7).fit_predict(tom)
    except Exception:
        return np.zeros(g.vcount(), dtype=int)


def alg_meanshift(g, seed=SEED):
    """Mean Shift：谱嵌入空间 + estimate_bandwidth"""
    from sklearn.cluster import estimate_bandwidth
    X = _spectral_feat(_tom_arr(g))
    from sklearn.metrics import silhouette_score
    best, best_sc = None, -1e9
    for q in (0.1, 0.2, 0.3, 0.5, 0.7):
        try:
            bw = estimate_bandwidth(X, quantile=q, n_samples=min(200, X.shape[0]),
                                    random_state=seed)
            if not bw or bw <= 0:
                continue
            lab = MeanShift(bandwidth=float(bw)).fit_predict(X)
        except Exception:
            continue
        k = len(set(lab))
        if k < 2 or k > max(15, X.shape[0] // 5):
            continue
        try:
            sc = silhouette_score(X, lab)
        except Exception:
            continue
        if sc > best_sc:
            best_sc, best = sc, lab
    if best is not None:
        return best
    try:
        bw = estimate_bandwidth(X, quantile=0.3, random_state=seed)
        return MeanShift(bandwidth=float(bw) if bw and bw > 0 else 1.0).fit_predict(X)
    except Exception:
        return np.zeros(X.shape[0], dtype=int)


# ---------- 矩阵分解 ----------
def alg_nmf(g, k=None, seed=SEED):
    tom = _tom_arr(g)
    tom = np.clip(tom, 0, None)
    if k is None:
        k = max(2, min(8, g.vcount() // 20))
    m = NMF(n_components=k, random_state=seed, init="nndsvda", max_iter=400)
    W = m.fit_transform(tom)
    return W.argmax(1)


# ---------- 重叠社区 ----------
def alg_link_communities(g, seed=SEED):
    """
    边社区检测（Link Communities, Ahn et al. 2010）
    在 line graph（边作为节点）上做社区检测，再映射回节点归属
    """
    import community as cl
    tom = _tom_arr(g)
    n = g.vcount()
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            if tom[i, j] > 0:
                G.add_edge(i, j, weight=float(tom[i, j]))
    if G.number_of_edges() < 5:
        return np.zeros(n, dtype=int)
    LG = nx.line_graph(G)
    if LG.number_of_nodes() < 5:
        return np.zeros(n, dtype=int)
    for u, v, d2 in LG.edges(data=True):
        w1 = G[u[0]][u[1]]["weight"]
        w2 = G[v[0]][v[1]]["weight"]
        d2["weight"] = float(w1 + w2)
    try:
        part = cl.best_partition(LG, weight="weight", random_state=seed)
    except Exception:
        return np.zeros(n, dtype=int)
    labels = np.full(n, -1, dtype=int)
    for (u, v), c in part.items():
        labels[u] = c
        labels[v] = c
    nxt = int(labels.max()) + 1 if (labels >= 0).any() else 0
    for i in range(n):
        if labels[i] < 0:
            labels[i] = nxt
            nxt += 1
    return labels


def alg_clique_percolation(g, k=3):
    """Clique Percolation（重叠社区）：取每个节点所属最大团"""
    tom = _tom_arr(g)
    n = g.vcount()
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            if tom[i, j] > 0.3:
                G.add_edge(i, j)
    labels = np.full(n, -1, dtype=int)
    try:
        comms = list(nx.community.k_clique_communities(G, k))
        for ci, c in enumerate(comms):
            for node in c:
                if labels[node] < 0:
                    labels[node] = ci
    except Exception:
        pass
    if (labels < 0).all():
        return np.zeros(n, dtype=int)
    # 未归属节点分到独立类
    nxt = labels.max() + 1
    for i in range(n):
        if labels[i] < 0:
            labels[i] = nxt
            nxt += 1
    return labels


# ---------- igraph 内置社区检测 ----------
def alg_fastgreedy(g):
    return np.array(g.community_fastgreedy(weights="weight").as_clustering().membership)


def alg_leading_eigen(g):
    return np.array(g.community_leading_eigenvector(weights="weight",
                                                    clusters=min(8, g.vcount() // 15 + 2)).membership)


def alg_edge_betweenness(g):
    return np.array(g.community_edge_betweenness(weights="weight").as_clustering().membership)


def alg_spinglass(g, seed=SEED):
    """Spin Glass：显式指定 spins 数量"""
    n = g.vcount()
    # 多组参数尝试，取模块数合理的解
    for sp in (n // 8, n // 12, 40, 25):
        for gam in (1.0, 2.0, 0.5):
            try:
                ig.set_random_number_generator(np.random.default_rng(seed))
                p = g.community_spinglass(weights="weight", spins=max(4, int(sp)),
                                          gamma=gam, gamma_minus=gam)
                k = len(p)
                if k >= 2:
                    return np.array(p.membership)
            except Exception:
                continue
    return np.zeros(n, dtype=int)


def alg_multilevel(g):
    return np.array(g.community_multilevel(weights="weight").membership)


def alg_optimal_modularity(g):
    try:
        return np.array(g.community_optimal_modularity(weights="weight").membership)
    except Exception:
        return np.zeros(g.vcount(), dtype=int)


# ---------- 扩展参数网格 ----------
def alg_leiden_g(g, gamma):
    import leidenalg as la
    return np.array(la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                      resolution_parameter=gamma, seed=SEED).membership)


def alg_cpm_g(g, gamma):
    import leidenalg as la
    return np.array(la.find_partition(g, la.CPMVertexPartition, weights="weight",
                                      resolution_parameter=gamma, seed=SEED).membership)


def alg_louvain_g(g, gamma):
    import community as community_louvain
    G = nx.Graph()
    tom = _tom_arr(g)
    n = g.vcount()
    for i in range(n):
        for j in range(i + 1, n):
            if tom[i, j] > 0:
                G.add_edge(i, j, weight=float(tom[i, j]))
    part = community_louvain.best_partition(G, weight="weight", resolution=gamma,
                                            random_state=SEED)
    return np.array([part.get(i, -1) for i in range(n)])


# ---------- 汇总注册表 ----------
NEW_ALGOS = {
    "Markov Stability": alg_markov_stability,
    "HDBSCAN": alg_hdbscan,
    "DBSCAN": alg_dbscan,
    "OPTICS": alg_optics,
    "Affinity Propagation": alg_affinity,
    "Mean Shift": alg_meanshift,
    "NMF": alg_nmf,
    "Link Communities": alg_link_communities,
    "Clique Percolation": alg_clique_percolation,
    "Fast Greedy": alg_fastgreedy,
    "Leading Eigenvector": alg_leading_eigen,
    "Edge Betweenness": alg_edge_betweenness,
    "Spin Glass": alg_spinglass,
    "Multilevel": alg_multilevel,
    "Optimal Modularity": alg_optimal_modularity,
    "Leiden γ=0.1": lambda g: alg_leiden_g(g, 0.1),
    "Leiden γ=10.0": lambda g: alg_leiden_g(g, 10.0),
    "CPM γ=0.20": lambda g: alg_cpm_g(g, 0.20),
    "Louvain γ=0.5": lambda g: alg_louvain_g(g, 0.5),
    "Louvain γ=2.0": lambda g: alg_louvain_g(g, 2.0),
}


if __name__ == "__main__":
    # 自检：构造已知 3 社区图，看各新算法能否恢复
    import sys
    sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
    from gwas_synthetic import make_block_tom
    from sklearn.metrics import adjusted_rand_score
    tom, true = make_block_tom(120, 5, 0.15, seed=42)
    g = ig.Graph.Weighted_Adjacency(tom.tolist(), mode=ig.ADJ_UNDIRECTED,
                                    attr="weight", loops=False)
    print("新算法自检（已知 5 社区，μ=0.15）:")
    for name, fn in NEW_ALGOS.items():
        try:
            lab = np.asarray(fn(g))
            ari = adjusted_rand_score(true, lab)
            print(f"  {name:<24} ARI={ari:+.3f}  模块数={len(set(lab[lab>=0]))}")
        except Exception as e:
            print(f"  {name:<24} ✗ {str(e)[:50]}")
