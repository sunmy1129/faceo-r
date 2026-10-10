#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证「模块度完全由权重驱动」这一发现的稳健性
测试: 不同阈值下，均匀权重的 Q 是否都崩塌
"""
import re
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import igraph as ig
import leidenalg as la
import gwas_full_analysis as GA
import gwas_full_analysis_v3 as V3

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
OUT = BASE / "output/core_results"
SEED = 42
THRESHOLDS = [50, 75, 85, 90, 95]


def build_g(tom, thr_pct):
    n = tom.shape[0]
    vals = tom[np.triu_indices(n, 1)]
    thr = np.percentile(vals, thr_pct)
    edges, w = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if tom[i, j] >= thr and tom[i, j] > 0:
                edges.append((i, j)); w.append(float(tom[i, j]))
    if len(edges) < 5:
        return None, None
    g = ig.Graph(n=n, edges=edges, directed=False)
    return g, np.array(w)


def leiden_q(g, w):
    p = la.find_partition(g, la.RBConfigurationVertexPartition,
                          weights=w, resolution_parameter=1.0, seed=SEED)
    return float(g.modularity(p, weights=w)), len(p)


def main(max_c=4, max_p=3):
    res = pd.read_csv(OUT / "full_eval_v3.csv")
    cohorts = sorted(res.gse.unique())[:max_c]
    rows = []
    for gse in cohorts:
        try:
            raw = pd.read_csv(BASE / f"qcdata/{gse}.csv", index_col=0,
                              quoting=3, low_memory=False)
            raw.index = [str(i).strip().strip('"').upper() for i in raw.index]
            keep = [re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(i)) is not None
                    for i in raw.index]
            df = V3.preprocess_robust(raw[keep])
        except Exception:
            continue
        go_all = V3.load_go_sets()
        paths = V3.select_pathways(go_all, set(df.index))
        for pw, pgenes in paths[:max_p]:
            genes = [g for g in pgenes if g in df.index]
            if not (60 <= len(genes) <= 250):
                continue
            tom = GA.compute_tom(df.loc[list(genes)])
            for thr in THRESHOLDS:
                g, w = build_g(tom, thr)
                if g is None:
                    continue
                q_w, m_w = leiden_q(g, w)                    # 用真实权重
                q_u, m_u = leiden_q(g, np.ones(len(w)))      # 均匀权重
                rows.append({"gse": gse, "pathway": pw[:26], "threshold": thr,
                             "Q_weighted": q_w, "Q_uniform": q_u,
                             "m_weighted": m_w, "m_uniform": m_u})
        print(f"  {gse} 完成", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "weight_dependence.csv", index=False)
    print(f"\n共 {len(d)} 组合 x 阈值")
    print("\n=== 各阈值：加权 vs 均匀权重的模块度 ===")
    agg = d.groupby("threshold").agg(
        Q_w=("Q_weighted", "mean"), Q_u=("Q_uniform", "mean"),
        m_w=("m_weighted", "mean"), m_u=("m_uniform", "mean")).round(4)
    agg["崩塌率"] = ((d.groupby("threshold").apply(
        lambda x: (x.Q_uniform < 0.05).mean())).round(3))
    print(agg.to_string())
    print(f"\n均匀权重下 Q<0.05 的比例: {(d.Q_uniform < 0.05).mean():.1%}")
    print(f"加权  平均 Q = {d.Q_weighted.mean():.4f}")
    print(f"均匀  平均 Q = {d.Q_uniform.mean():.4f}")
    print(f"\n结论: 模块度{'完全依赖权重' if d.Q_uniform.mean() < 0.05 else '部分依赖权重'}")


if __name__ == "__main__":
    main()
