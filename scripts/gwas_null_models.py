#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零模型检验（H5 假设）：真实网络的模块结构能否被随机模型复现？
零模型 A：度保持随机重连（保留度分布，破坏高阶结构）
零模型 B：表达矩阵基因级置换（破坏共表达结构）
输出：null_models.csv
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import igraph as ig
import leidenalg as la

import gwas_full_analysis as GA

warnings.filterwarnings("ignore")
BASE = Path("/Users/sunmy/Documents/QF/GWAS")
HUB, DATA = BASE / "output/hubs", BASE / "qcdata"
OUT = BASE / "output/core_results"
SEED = 42
N_NULL = 30


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def q_of(tom, gamma=1.0):
    """在给定 TOM 上求 Leiden 最优划分的模块度"""
    g = GA.to_ig(tom, [f"g{i}" for i in range(tom.shape[0])])
    p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                          resolution_parameter=gamma, seed=SEED)
    if len(p) < 2:
        return np.nan, len(p)
    return float(g.modularity(p.membership, weights="weight")), len(p)


def degree_preserving_rewire(tom, n_swap=None, seed=SEED):
    """度保持随机重连（igraph C 实现，快）"""
    import random
    n = tom.shape[0]
    rng = np.random.default_rng(seed)
    thr = np.percentile(tom[tom > 0], 75) if (tom > 0).any() else 0
    A = (tom > thr).astype(int)
    np.fill_diagonal(A, 0)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]
    if len(edges) < 10:
        return tom.copy()
    n_edges = len(edges)
    el = []
    for (i, j) in edges:
        el += [i, j]
    g = ig.Graph(n=n, edges=[(el[k], el[k+1]) for k in range(0, len(el), 2)])
    random.seed(seed)
    try:
        g.rewire(n=n_swap or 3 * n_edges, mode="simple")
    except Exception:
        pass
    w_pool = tom[tom > 0]
    null = np.zeros((n, n))
    for e in g.get_edgelist():
        i, j = e
        v = float(rng.choice(w_pool)) if len(w_pool) else 0.5
        null[i, j] = null[j, i] = v
    return null


def expression_permutation(df_sub, seed=SEED):
    """基因级置换：每个基因的表达向量独立打乱后重算 TOM"""
    rng = np.random.default_rng(seed)
    vals = df_sub.values.copy()
    for i in range(vals.shape[0]):
        rng.shuffle(vals[i])
    dfp = pd.DataFrame(vals, index=df_sub.index, columns=df_sub.columns)
    return GA.compute_tom(dfp)


def main(n_null=N_NULL):
    metas = sorted(HUB.glob("*_meta.json"))
    rows = []
    t0 = time.time()
    for mi, mf in enumerate(metas, 1):
        meta = json.load(open(mf))
        gse, pw = meta["gse"], meta["pathway"]
        df = GA.preprocess(pd.read_csv(DATA / f"{gse}.csv", index_col=0))
        avail = [g for g in meta["genes"] if g in df.index]
        if len(avail) < 20:
            continue
        df_sub = df.loc[avail]
        tom = GA.compute_tom(df_sub)

        q_real, n_real = q_of(tom)

        # 零模型 A：度保持重连
        qA = []
        for s in range(n_null):
            tn = degree_preserving_rewire(tom, seed=SEED + s)
            q, _ = q_of(tn)
            if q == q:
                qA.append(q)
        # 零模型 B：表达置换
        qB = []
        for s in range(n_null):
            tn = expression_permutation(df_sub, seed=SEED + s)
            q, _ = q_of(tn)
            if q == q:
                qB.append(q)

        qA, qB = np.array(qA), np.array(qB)
        rows.append({
            "gse": gse, "pathway": pw, "n_genes": len(avail),
            "Q_real": q_real, "n_mod_real": n_real,
            "Q_nullA_mean": qA.mean() if len(qA) else np.nan,
            "Q_nullA_sd": qA.std() if len(qA) else np.nan,
            "Q_nullA_p95": np.percentile(qA, 95) if len(qA) else np.nan,
            "z_nullA": (q_real - qA.mean()) / (qA.std() + 1e-9) if len(qA) else np.nan,
            "Q_nullB_mean": qB.mean() if len(qB) else np.nan,
            "Q_nullB_sd": qB.std() if len(qB) else np.nan,
            "z_nullB": (q_real - qB.mean()) / (qB.std() + 1e-9) if len(qB) else np.nan,
            "p_nullA": float((qA >= q_real).mean()) if len(qA) else np.nan,
            "p_nullB": float((qB >= q_real).mean()) if len(qB) else np.nan,
        })
        if mi % 5 == 0 or mi == len(metas):
            log(f"({mi}/{len(metas)}) {gse} | z_A={rows[-1]['z_nullA']:.1f} "
                f"z_B={rows[-1]['z_nullB']:.1f}")

    df_res = pd.DataFrame(rows)
    df_res.to_csv(OUT / "null_models.csv", index=False)
    log(f"✅ 已保存 null_models.csv ({len(df_res)} 行, {time.time()-t0:.0f}s)")

    print()
    print("="*90)
    print("零模型检验（H5）：真实网络模块度 vs 随机模型")
    print("="*90)
    print(f"  零模型 A（度保持重连）: 平均 z = {df_res['z_nullA'].mean():.2f}, "
          f"显著比例 (p<0.05) = {(df_res['p_nullA'] < 0.05).mean()*100:.0f}%")
    print(f"  零模型 B（表达置换）  : 平均 z = {df_res['z_nullB'].mean():.2f}, "
          f"显著比例 (p<0.05) = {(df_res['p_nullB'] < 0.05).mean()*100:.0f}%")
    return df_res


if __name__ == "__main__":
    main()
