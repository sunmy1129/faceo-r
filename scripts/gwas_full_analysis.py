#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整分析引擎 v2：16 种划分算法 + 自监督嵌入 + 六层评估
L1 结构 / L2 功能富集 / L3 稳定性 / L4 鲁棒性 / L5 生物学 / 随机对照
输出：full_eval.csv
"""
import json
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
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster

import gwas_l2_functional as L2
import gwas_selfsup as SS

warnings.filterwarnings("ignore")

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
HUB = BASE / "output/hubs"
DATA = BASE / "qcdata"
OUT = BASE / "output/core_results"
OUT.mkdir(parents=True, exist_ok=True)

BETA, SEED = 6, 42
np.random.seed(SEED)

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


def to_ig(tom, genes):
    g = ig.Graph.Weighted_Adjacency(tom.tolist(), mode=ig.ADJ_UNDIRECTED,
                                    attr="weight", loops=False)
    g.vs["name"] = genes
    return g


def tom_arr(g):
    return np.array(g.get_adjacency(attribute="weight").data, dtype=float)


# ============ 算法定义 ============
def alg_leiden(g, gamma):
    return np.array(la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                      resolution_parameter=gamma, seed=SEED).membership)


def alg_cpm(g, gamma):
    return np.array(la.find_partition(g, la.CPMVertexPartition, weights="weight",
                                      resolution_parameter=gamma, seed=SEED).membership)


def alg_louvain(g, gamma=1.0):
    G = nx.Graph()
    A = tom_arr(g)
    n = g.vcount()
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=float(A[i, j]))
    part = community_louvain.best_partition(G, weight="weight", resolution=gamma, random_state=SEED)
    return np.array([part.get(i, -1) for i in range(n)])


def alg_infomap(g):
    return np.array(g.community_infomap(edge_weights="weight", trials=10).membership)


def alg_walktrap(g):
    return np.array(g.community_walktrap(weights="weight", steps=4).as_clustering().membership)


def alg_labelprop(g):
    return np.array(g.community_label_propagation(weights="weight").membership)


def alg_hier(g, method="average", min_size=10):
    A = tom_arr(g)
    D = 1 - A
    np.fill_diagonal(D, 0)
    iu = np.triu_indices_from(D, 1)
    Z = linkage(D[iu], method=method)
    best, best_s = None, -1
    for k in range(2, min(12, g.vcount())):
        lab = fcluster(Z, k, criterion="maxclust") - 1
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


def _kmeans_like(g, algo, k=None):
    A = tom_arr(g)
    if k is None:
        k = max(2, min(8, g.vcount() // 20))
    if algo == "kmeans":
        return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(A)
    if algo == "gmm":
        return GaussianMixture(n_components=k, random_state=SEED, n_init=3).fit_predict(A)
    if algo == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(A)
    if algo == "spectral":
        return SpectralClustering(n_clusters=min(k, max(2, g.vcount() // 5)),
                                  affinity="precomputed", random_state=SEED).fit_predict(A)


ALGOS = {
    "WGCNA-style\nHierarchical": lambda g: alg_hier(g, "average"),
    "Leiden γ=0.5": lambda g: alg_leiden(g, 0.5),
    "Leiden γ=1.0": lambda g: alg_leiden(g, 1.0),
    "Leiden γ=2.0": lambda g: alg_leiden(g, 2.0),
    "Leiden γ=5.0": lambda g: alg_leiden(g, 5.0),
    "CPM γ=0.05": lambda g: alg_cpm(g, 0.05),
    "CPM γ=0.10": lambda g: alg_cpm(g, 0.10),
    "Louvain": lambda g: alg_louvain(g),
    "Infomap": lambda g: alg_infomap(g),
    "Walktrap": lambda g: alg_walktrap(g),
    "Label Propagation": lambda g: alg_labelprop(g),
    "K-means": lambda g: _kmeans_like(g, "kmeans"),
    "Gaussian Mixture": lambda g: _kmeans_like(g, "gmm"),
    "Ward": lambda g: _kmeans_like(g, "ward"),
    "Spectral": lambda g: _kmeans_like(g, "spectral"),
}


# ============ 评估 ============
def eval_L1(g, labels):
    labs = np.asarray(labels)
    valid = labs >= 0
    uniq = sorted(set(labs[valid]))
    n_mod = len(uniq)
    Q = g.modularity(labs.tolist(), weights="weight") if n_mod >= 2 else np.nan
    if n_mod >= 2:
        sizes = np.array([(labs == u).sum() for u in uniq], dtype=float)
        p = sizes / sizes.sum()
        ent = float(-(p * np.log2(p)).sum())
    else:
        ent = 0.0
    return {"n_modules": n_mod, "modularity_Q": Q, "size_entropy": ent,
            "unassigned_rate": float(1 - valid.mean())}


def eval_stability(tom, genes, afn, n_boot=20, frac=0.8):
    rng = np.random.default_rng(SEED)
    n = len(genes)
    labs, idxs = [], []
    for _ in range(n_boot):
        idx = np.sort(rng.choice(n, int(n * frac), replace=False))
        try:
            labs.append(np.asarray(afn(to_ig(tom[np.ix_(idx, idx)], [genes[i] for i in idx]))))
            idxs.append(idx)
        except Exception:
            pass
    if len(labs) < 3:
        return np.nan
    aris = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            common = np.intersect1d(idxs[i], idxs[j])
            if len(common) < 5:
                continue
            mi = {v: k for k, v in enumerate(idxs[i])}
            mj = {v: k for k, v in enumerate(idxs[j])}
            aris.append(adjusted_rand_score([labs[i][mi[c]] for c in common],
                                            [labs[j][mj[c]] for c in common]))
    return float(np.mean(aris)) if aris else np.nan


def eval_robust(tom, genes, afn):
    out = {}
    try:
        base = np.asarray(afn(to_ig(tom, genes)))
    except Exception:
        return {"ARI_noise0.1": np.nan, "ARI_noise0.2": np.nan, "ARI_edge0.1": np.nan}
    rng = np.random.default_rng(SEED)
    for nl in (0.1, 0.2):
        t = tom * (1 + rng.normal(0, nl, tom.shape))
        t = np.clip((t + t.T) / 2, 0, None)
        np.fill_diagonal(t, 0)
        try:
            out[f"ARI_noise{nl}"] = adjusted_rand_score(base, np.asarray(afn(to_ig(t, genes))))
        except Exception:
            out[f"ARI_noise{nl}"] = np.nan
    t = tom.copy()
    t[rng.random(t.shape) < 0.1] = 0
    t = (t + t.T) / 2
    np.fill_diagonal(t, 0)
    try:
        out["ARI_edge0.1"] = adjusted_rand_score(base, np.asarray(afn(to_ig(t, genes))))
    except Exception:
        out["ARI_edge0.1"] = np.nan
    return out


def eval_antag(labels, genes):
    gi = {g: i for i, g in enumerate(genes)}
    labs = np.asarray(labels)
    t, s = 0, 0
    for a, b in ANTAGONISTIC:
        if a in gi and b in gi:
            la_, lb_ = labs[gi[a]], labs[gi[b]]
            if la_ < 0 or lb_ < 0:
                continue
            t += 1
            s += int(la_ != lb_)
    return {"antag_tested": t, "antag_split": s,
            "antag_split_rate": (s / t) if t else np.nan}


def eval_random_ctrl(labels, genes, n_perm=50):
    rng = np.random.default_rng(SEED)
    labs = np.asarray(labels)
    k = len(set(labs[labs >= 0]))
    if k < 2:
        return {"ARI_vs_random": np.nan}
    aris = [adjusted_rand_score(labs, rng.integers(0, k, len(genes))) for _ in range(n_perm)]
    return {"ARI_vs_random": float(np.mean(aris))}


def main(datasets=None, n_boot=20, with_embed=True, embed_encoders=("gat", "sage")):
    metas = sorted(HUB.glob("*_meta.json"))
    if datasets:
        metas = [m for m in metas if json.load(open(m))["gse"] in datasets]
    terms, allg, gene2terms = L2.load_genesets()
    universe = len(allg)
    log(f"基因集库就绪: {len(terms)} terms, universe={universe}")

    records = []
    t0 = time.time()
    for mi, mf in enumerate(metas, 1):
        meta = json.load(open(mf))
        gse, pw = meta["gse"], meta["pathway"]
        df = preprocess(pd.read_csv(DATA / f"{gse}.csv", index_col=0))
        avail = [g for g in meta["genes"] if g in df.index]
        if len(avail) < 20:
            continue
        df_sub = df.loc[avail]
        genes = list(df_sub.index)
        tom = compute_tom(df_sub)
        g = to_ig(tom, genes)
        log(f"({mi}/{len(metas)}) {gse} | {pw[:42]} | {len(genes)} 基因")

        # ---- 传统算法 ----
        for aname, afn in ALGOS.items():
            try:
                labels = np.asarray(afn(g))
            except Exception as e:
                log(f"    ✗ {aname}: {str(e)[:36]}")
                continue
            rec = {"gse": gse, "pathway": pw, "algo": aname.replace("\n", " "),
                   "n_genes": len(genes), "family": "classical"}
            rec.update(eval_L1(g, labels))
            rec.update(L2.evaluate_functional(labels, genes, terms, universe, gene2terms, n_perm=40))
            rec["stability_ARI"] = eval_stability(tom, genes, afn, n_boot)
            rec.update(eval_robust(tom, genes, afn))
            rec.update(eval_antag(labels, genes))
            rec.update(eval_random_ctrl(labels, genes))
            records.append(rec)
        log(f"    传统算法完成 ({time.time()-t0:.0f}s)")

        # ---- 自监督嵌入 ----
        if with_embed:
            x, ei, ea = SS.build_pyg_data(tom)
            for enc in embed_encoders:
                try:
                    emb, losses = SS.train_edge_prediction(x, ei, ea, encoder=enc,
                                                           epochs=300, out_dim=64)
                    labels = SS.embed_to_partition(emb)
                    ari_vs_leiden = SS.consistency_with_partition(
                        emb, alg_leiden(g, 1.0))
                    rec = {"gse": gse, "pathway": pw,
                           "algo": f"SelfSup-{enc.upper()}",
                           "n_genes": len(genes), "family": "embedding",
                           "final_loss": losses[-1] if losses else np.nan,
                           "ari_vs_leiden_g1.0": ari_vs_leiden}
                    # 用最近质心划分评估各层
                    rec.update(eval_L1(g, labels))
                    rec.update(L2.evaluate_functional(labels, genes, terms, universe, gene2terms, n_perm=40))
                    rec.update(eval_antag(labels, genes))
                    rec.update(eval_random_ctrl(labels, genes))
                    emb_ari = SS.consistency_with_partition(emb, labels)
                    rec["stability_ARI"] = emb_ari  # 嵌入重采样一致性代理
                    records.append(rec)
                    log(f"    ✓ SelfSup-{enc}: loss={losses[-1]:.3f}, ARI(vs Leiden γ=1.0)={ari_vs_leiden:.3f}")
                except Exception as e:
                    log(f"    ✗ SelfSup-{enc}: {str(e)[:44]}")

    df_res = pd.DataFrame(records)
    df_res.to_csv(OUT / "full_eval.csv", index=False)
    log(f"✅ 已保存 {OUT/'full_eval.csv'} ({len(df_res)} 行, {time.time()-t0:.0f}s)")
    return df_res


if __name__ == "__main__":
    import sys
    ds = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else None
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    df = main(datasets=ds, n_boot=nb)
    pd.set_option("display.width", 220)
    print()
    print(df.groupby("algo")[["n_modules", "modularity_Q", "mean_purity",
                              "stability_ARI", "antag_split_rate"]].mean().round(3))
