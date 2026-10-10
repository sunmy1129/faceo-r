#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
替代方案测试：
1) 最优 γ 是否与通路规模/密度相关（简单分组规则是否可行）
2) 多分辨率共识聚类（不预测 γ，而是融合多尺度）是否优于最佳单 γ
"""
import sys
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score as ari

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import gwas_full_analysis as GA
import gwas_full_analysis_v3 as V3
import gwas_l2_functional as L2

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
RES = BASE / "output/core_results/full_eval_v3.csv"
GAMMAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def load_cohort(gse):
    raw = pd.read_csv(BASE / f"qcdata/{gse}.csv", index_col=0, quoting=3,
                      low_memory=False)
    raw.index = [str(i).strip().strip('"').upper() for i in raw.index]
    keep = [re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(i)) is not None
            for i in raw.index]
    raw = raw[keep]
    return V3.preprocess_robust(raw)


def test1():
    """最优 γ 与结构特征的相关性"""
    df = pd.read_csv(RES)
    df["algo"] = df["algo"].astype(str)
    lg = df[df.algo.str.startswith("Leiden γ=")].copy()
    lg["gamma"] = lg.algo.str.extract(r"γ=([0-9.]+)").astype(float)
    rows = []
    for (gse, pw), sub in lg.groupby(["gse", "pathway"]):
        sub = sub.set_index("gamma").sort_index()
        if len(sub) < 6:
            continue
        rows.append({"n_genes": sub.n_genes.iloc[0],
                     "opt_gamma": sub.coh_z_mean.idxmax(),
                     "Q_max": sub.modularity_Q.max()})
    d = pd.DataFrame(rows)
    print("=== 测试 1：最优 γ 与通路规模的相关性 ===")
    rho, p = spearmanr(d["n_genes"], d["opt_gamma"])
    print(f"  Spearman(基因数, 最优γ) = {rho:+.3f}, p = {p:.3f}")
    print(f"  {'✅ 有显著相关，分组规则可行' if p < 0.05 else '❌ 无显著相关，分组规则不可行'}")
    # 分箱看
    d["bin"] = pd.cut(d.n_genes, [0, 100, 150, 200, 1000], labels=["<100", "100-150", "150-200", ">200"])
    print("\n  按规模分箱的最优 γ 分布:")
    for b, sub in d.groupby("bin"):
        if len(sub):
            vc = sub.opt_gamma.value_counts()
            top = vc.index[0]
            print(f"    {str(b):<9} n={len(sub):<3} 最优γ众数={top:<5} 占 {vc.iloc[0]/len(sub)*100:.0f}%")
    return d


def test2(max_combos=12):
    """多分辨率共识聚类"""
    print("\n=== 测试 2：多分辨率共识聚类 vs 最佳单 γ ===")
    terms, allg, g2t = L2.load_genesets()
    cohorts = sorted(pd.read_csv(RES).gse.unique())
    results = []
    for gse in cohorts[:4]:
        try:
            df = load_cohort(gse)
        except Exception:
            continue
        go = V3.load_go_sets()
        paths = V3.select_pathways(go, set(df.index))
        for pw_name, pw_genes in paths[:3]:
            genes = [g for g in pw_genes if g in df.index]
            if len(genes) < 40:
                continue
            if len(genes) > 200:
                v = df.loc[genes].var(axis=1).sort_values(ascending=False)
                genes = list(v.index[:200])
            tom = GA.compute_tom(df.loc[genes])
            g = GA.to_ig(tom, genes)
            # 各 γ 的划分
            labs = {}
            for gm in GAMMAS:
                fn = GA.ALGOS.get(f"Leiden γ={gm}")
                if fn is None:
                    continue
                labs[gm] = np.asarray(fn(g))
            if len(labs) < 4:
                continue
            n = len(genes)
            # 共识矩阵：基因对同模块频率
            C = np.zeros((n, n))
            for lb in labs.values():
                same = (lb[:, None] == lb[None, :]).astype(float)
                C += same
            C /= len(labs)
            # 用 TOM 做底、共识做权重的层次聚类
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            D = 1 - C
            np.fill_diagonal(D, 0)
            Z = linkage(squareform(D, checks=False), method="average")
            k_ref = len(set(labs[1.0]))
            lab_cons = fcluster(Z, t=k_ref, criterion="maxclust") - 1
            # 评价功能一致性
            z_cons, _ = L2.coherence_z(list(genes), g2t, list(genes), n_perm=15)
            z_best = max(L2.coherence_z(list(genes), g2t, list(genes), n_perm=15)[0]
                         if False else -9 for _ in [0])
            # 各单 γ 的功能一致性（用已存结果）
            sub = pd.read_csv(RES)
            sub["algo"] = sub["algo"].astype(str)
            row = sub[(sub.gse == gse) & (sub.pathway == pw_name)]
            z_per = {gm: row[row.algo == f"Leiden γ={gm}"]["coh_z_mean"].mean()
                     for gm in GAMMAS}
            z_best = max(z_per.values())
            results.append({"gse": gse, "pathway": pw_name[:30],
                            "z_consensus": z_cons, "z_best_single": z_best,
                            "delta": z_cons - z_best,
                            "k": k_ref})
    if not results:
        print("  无可用组合")
        return None
    r = pd.DataFrame(results)
    print(r.to_string(index=False))
    win = (r.delta > 0).sum()
    print(f"\n  共识优于最佳单γ: {win}/{len(r)} 组合")
    print(f"  平均增益: {r.delta.mean():+.3f}")
    return r


if __name__ == "__main__":
    test1()
    test2()
