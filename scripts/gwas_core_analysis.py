#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心分析引擎：在同一条 GO 通路子网上对比 12 种划分算法
评估层：L1 结构 / L3 稳定性 / L4 鲁棒性 / L5 生物学（拮抗基因对）/ 空白对照
输出：results/core_eval.csv（长表，可继续接统计与绘图）
"""
import json
import glob
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import igraph as ig
import leidenalg as la
import networkx as nx
import community as community_louvain
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
HUB = BASE / "output/hubs"
DATA = BASE / "qcdata"
OUT = BASE / "output/core_results"
OUT.mkdir(parents=True, exist_ok=True)

BETA = 6
SEED = 42
np.random.seed(SEED)

# ---------- 拮抗基因对（L5 生物学验证） ----------
ANTAGONISTIC = [
    ("BAX", "BCL2"), ("CASP3", "XIAP"), ("CASP8", "CFLAR"), ("BID", "BCL2L1"),
    ("NFKB1", "NFKBIA"), ("NFKB1", "TNFAIP3"), ("TNF", "IL10"), ("IL6", "IL10"),
    ("CXCL8", "IL10"), ("IFNG", "IL4"), ("IFNG", "IL10"), ("IL1B", "IL1RN"),
    ("TP53", "MDM2"), ("CCND1", "CDKN1A"), ("MYC", "CDKN2A"), ("BCL2L1", "BAK1"),
    ("MAPK1", "DUSP1"), ("STAT1", "SOCS1"), ("STAT3", "SOCS3"), ("AKT1", "PTEN"),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def preprocess(df):
    """与原 pipeline 保持一致的预处理"""
    df = df.copy()
    df = df.fillna(df.median(axis=1), axis=0)
    if df.max().max() > 100:
        df = np.log2(df + 1)
    keep = (df > np.log2(10)).sum(axis=1) > (df.shape[1] * 0.5)
    return df.loc[keep]


def compute_tom(df_sub, beta=BETA):
    data = df_sub.values.astype(np.float32)
    corr = np.corrcoef(data)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 0)
    adj = np.abs(corr) ** beta
    k = adj.sum(axis=1)
    l_ij = adj @ adj
    denom = np.minimum(k[:, None], k[None, :]) + 1 - adj
    denom[denom == 0] = 1e-10
    tom = (l_ij + adj) / denom
    tom = np.nan_to_num(tom, nan=0.0, posinf=0.0, neginf=0.0)
    tom = (tom + tom.T) / 2
    np.fill_diagonal(tom, 0)
    return tom


def tom_to_igraph(tom, genes):
    g = ig.Graph.Weighted_Adjacency(tom.tolist(), mode=ig.ADJ_UNDIRECTED, attr="weight", loops=False)
    g.vs["name"] = genes
    return g


# ---------------- 12 种算法 ----------------
def alg_leiden(g, gamma=1.0, seed=SEED):
    p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                          resolution_parameter=gamma, seed=seed)
    return np.array(p.membership)


def alg_cpm(g, gamma=0.05, seed=SEED):
    p = la.find_partition(g, la.CPMVertexPartition, weights="weight",
                          resolution_parameter=gamma, seed=seed)
    return np.array(p.membership)


def alg_louvain(g, seed=SEED, gamma=1.0):
    G = nx.Graph()
    for u, v in g.get_edgelist():
        G.add_edge(u, v, weight=tom_weight(g, u, v))
    part = community_louvain.best_partition(G, weight="weight", resolution=gamma, random_state=seed)
    return np.array([part.get(i, -1) for i in range(g.vcount())])


def tom_weight(g, u, v):
    eid = g.get_eid(u, v, error=False)
    return g.es[eid]["weight"] if eid >= 0 else 0.0


def alg_infomap(g, seed=SEED, trials=10):
    p = g.community_infomap(edge_weights="weight", trials=trials)
    return np.array(p.membership)


def alg_walktrap(g, steps=4):
    p = g.community_walktrap(weights="weight", steps=steps).as_clustering()
    return np.array(p.membership)


def alg_labelprop(g, seed=SEED):
    p = g.community_label_propagation(weights="weight", initial=None, fixed=None)
    return np.array(p.membership)


def alg_hierarchical(g, method="average", k=None, min_size=10):
    """层次聚类 + 固定切割（模拟 WGCNA 的树切割框架）"""
    diss = 1 - np.array(g.get_adjacency(attribute="weight").data, dtype=float)
    np.fill_diagonal(diss, 0)
    Z = linkage(diss[np.triu_indices_from(diss, 1)], method=method)
    if k is None:
        best, best_s = None, -1
        for kk in range(2, min(12, g.vcount())):
            lab = fcluster(Z, kk, criterion="maxclust") - 1
            sizes = np.bincount(lab)
            if (sizes < min_size).any():
                continue
            bal = 1.0 / (1.0 + np.std(sizes) / max(1e-9, np.mean(sizes)))
            s = len(sizes) * bal
            if s > best_s:
                best_s, best = s, lab
        if best is None:
            best = fcluster(Z, min(3, g.vcount() - 1), criterion="maxclust") - 1
        return best.astype(int)
    return (fcluster(Z, k, criterion="maxclust") - 1).astype(int)


def alg_sbm(g, seed=SEED):
    """随机块模型（igraph 内置）"""
    try:
        p = g.community_leiden(objective_function="modularity", weights="weight", n_iterations=3)
        return np.array(p.membership)
    except Exception:
        return np.zeros(g.vcount(), dtype=int)


def alg_kmeans(g, k=None, seed=SEED):
    tom = np.array(g.get_adjacency(attribute="weight").data, dtype=float)
    if k is None:
        k = max(2, min(8, g.vcount() // 20))
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(tom)


def alg_gmm(g, k=None, seed=SEED):
    tom = np.array(g.get_adjacency(attribute="weight").data, dtype=float)
    if k is None:
        k = max(2, min(8, g.vcount() // 20))
    return GaussianMixture(n_components=k, random_state=seed, n_init=3).fit_predict(tom)


def alg_ward(g, k=None):
    tom = np.array(g.get_adjacency(attribute="weight").data, dtype=float)
    if k is None:
        k = max(2, min(8, g.vcount() // 20))
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(tom)


def alg_spectral(g, k=None, seed=SEED):
    tom = np.array(g.get_adjacency(attribute="weight").data, dtype=float)
    if k is None:
        k = max(2, min(8, g.vcount() // 20))
    return SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed).fit_predict(tom)


ALGOS = {
    "Leiden_g1.0": lambda g: alg_leiden(g, 1.0),
    "Leiden_g2.0": lambda g: alg_leiden(g, 2.0),
    "Leiden_g5.0": lambda g: alg_leiden(g, 5.0),
    "CPM_g0.05": lambda g: alg_cpm(g, 0.05),
    "CPM_g0.10": lambda g: alg_cpm(g, 0.10),
    "Louvain": lambda g: alg_louvain(g),
    "Infomap": lambda g: alg_infomap(g),
    "Walktrap": lambda g: alg_walktrap(g),
    "LabelProp": lambda g: alg_labelprop(g),
    "HierAvg": lambda g: alg_hierarchical(g, "average"),
    "KMeans": lambda g: alg_kmeans(g),
    "GMM": lambda g: alg_gmm(g),
    "Ward": lambda g: alg_ward(g),
    "Spectral": lambda g: alg_spectral(g),
}

# ---------------- 评估指标 ----------------
def eval_L1(g, labels):
    """结构层"""
    labs = np.asarray(labels)
    valid = labs >= 0
    unassigned = 1.0 - valid.mean()
    uniq = [x for x in set(labs[valid]) if x >= 0]
    n_mod = len(uniq)
    if n_mod < 2:
        Q = np.nan
    else:
        Q = g.modularity(labs.tolist(), weights="weight")
    # 模块大小熵
    if n_mod >= 2:
        sizes = np.array([(labs == u).sum() for u in uniq], dtype=float)
        p = sizes / sizes.sum()
        ent = float(-(p * np.log2(p)).sum())
    else:
        ent = 0.0
    return {"n_modules": n_mod, "modularity_Q": Q, "size_entropy": ent,
            "unassigned_rate": unassigned}


def eval_L3_stability(tom, genes, algo_fn, n_boot=20, frac=0.8, seed=SEED):
    """稳定性：bootstrap 重采样后的平均 ARI"""
    rng = np.random.default_rng(seed)
    n = len(genes)
    labels_list = []
    idx_list = []
    for b in range(n_boot):
        idx = rng.choice(n, size=int(n * frac), replace=False)
        idx = np.sort(idx)
        sub = tom[np.ix_(idx, idx)]
        try:
            g = tom_to_igraph(sub, [genes[i] for i in idx])
            lab = algo_fn(g)
            labels_list.append(np.asarray(lab))
            idx_list.append(idx)
        except Exception:
            continue
    if len(labels_list) < 3:
        return np.nan, 0
    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            common = np.intersect1d(idx_list[i], idx_list[j])
            if len(common) < 5:
                continue
            mi = {v: k for k, v in enumerate(idx_list[i])}
            mj = {v: k for k, v in enumerate(idx_list[j])}
            a = np.array([labels_list[i][mi[c]] for c in common])
            b = np.array([labels_list[j][mj[c]] for c in common])
            aris.append(adjusted_rand_score(a, b))
    return (float(np.mean(aris)) if aris else np.nan), len(labels_list)


def eval_L4_robustness(tom, genes, algo_fn, seed=SEED, noise_levels=(0.1, 0.2), edge_fracs=(0.1,)):
    """鲁棒性：加噪 / 删边后的 ARI（相对基线）"""
    out = {}
    try:
        g0 = tom_to_igraph(tom, genes)
        base = np.asarray(algo_fn(g0))
    except Exception:
        return out
    rng = np.random.default_rng(seed)
    for nl in noise_levels:
        t = tom * (1 + rng.normal(0, nl, tom.shape))
        t = np.clip((t + t.T) / 2, 0, None)
        np.fill_diagonal(t, 0)
        try:
            lab = np.asarray(algo_fn(tom_to_igraph(t, genes)))
            out[f"ARI_noise{nl}"] = adjusted_rand_score(base, lab)
        except Exception:
            out[f"ARI_noise{nl}"] = np.nan
    for ef in edge_fracs:
        t = tom.copy()
        mask = rng.random(t.shape) < ef
        t[mask] = 0
        t = (t + t.T) / 2
        np.fill_diagonal(t, 0)
        try:
            lab = np.asarray(algo_fn(tom_to_igraph(t, genes)))
            out[f"ARI_edge{ef}"] = adjusted_rand_score(base, lab)
        except Exception:
            out[f"ARI_edge{ef}"] = np.nan
    return out


def eval_L5_antagonists(labels, genes):
    """生物学层：拮抗基因对被拆分的比例"""
    gene_idx = {g: i for i, g in enumerate(genes)}
    labs = np.asarray(labels)
    tested, split = 0, 0
    for a, b in ANTAGONISTIC:
        if a in gene_idx and b in gene_idx:
            la_, lb_ = labs[gene_idx[a]], labs[gene_idx[b]]
            if la_ < 0 or lb_ < 0:
                continue
            tested += 1
            if la_ != lb_:
                split += 1
    return {"antag_tested": tested,
            "antag_split": split,
            "antag_split_rate": (split / tested) if tested else np.nan}


def eval_control(labels, genes, n_perm=50, seed=SEED):
    """空白对照：与随机划分的 ARI（检验划分是否非随机）"""
    rng = np.random.default_rng(seed)
    labs = np.asarray(labels)
    k = len(set(labs[labs >= 0]))
    if k < 2:
        return {"ARI_vs_random": np.nan}
    aris = []
    for _ in range(n_perm):
        rnd = rng.integers(0, k, size=len(genes))
        aris.append(adjusted_rand_score(labs, rnd))
    return {"ARI_vs_random": float(np.mean(aris))}


# ---------------- 主流程 ----------------
def main(datasets=None, algos=None, n_boot=20):
    metas = sorted(HUB.glob("*_meta.json"))
    if datasets:
        metas = [m for m in metas if json.load(open(m))["gse"] in datasets]
    records = []
    t0 = time.time()
    for mi, mf in enumerate(metas, 1):
        meta = json.load(open(mf))
        gse, pw = meta["gse"], meta["pathway"]
        genes_meta = meta["genes"]
        log(f"({mi}/{len(metas)}) {gse} | {pw[:45]}")

        df = pd.read_csv(DATA / f"{gse}.csv", index_col=0)
        df = preprocess(df)
        avail = [g for g in genes_meta if g in df.index]
        if len(avail) < 20:
            log(f"    跳过：可用基因仅 {len(avail)}")
            continue
        df_sub = df.loc[avail]
        genes = list(df_sub.index)
        tom = compute_tom(df_sub)
        g = tom_to_igraph(tom, genes)

        for aname, afn in (algos or ALGOS).items():
            try:
                labels = np.asarray(afn(g))
            except Exception as e:
                log(f"    {aname} 失败: {str(e)[:40]}")
                continue
            rec = {"gse": gse, "pathway": pw, "algo": aname, "n_genes": len(genes)}
            rec.update(eval_L1(g, labels))
            stab, n_ok = eval_L3_stability(tom, genes, afn, n_boot=n_boot)
            rec["stability_ARI"] = stab
            rec["n_boot_ok"] = n_ok
            rec.update(eval_L4_robustness(tom, genes, afn))
            rec.update(eval_L5_antagonists(labels, genes))
            rec.update(eval_control(labels, genes))
            records.append(rec)
        log(f"    完成 {len(records)} 条记录 ({time.time()-t0:.0f}s)")

    df_res = pd.DataFrame(records)
    out_csv = OUT / "core_eval.csv"
    df_res.to_csv(out_csv, index=False)
    log(f"已保存 {out_csv}  ({len(df_res)} 行)")
    return df_res


if __name__ == "__main__":
    import sys
    ds = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    df = main(datasets=ds)
    print(df.groupby("algo")[["n_modules", "modularity_Q", "stability_ARI",
                              "antag_split_rate", "unassigned_rate"]].mean().round(3))
