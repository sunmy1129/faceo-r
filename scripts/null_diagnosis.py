#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 2：解释零模型负 z（为什么观测模块度低于随机重连）
假设：
  H-a 度分布异质性 —— TOM 图高度异质，重连后仍保持度分布但边权重重排 -> Q 变化
  H-b 权重-结构耦合 —— 观测图中高权重边集中于局部；重连打散该耦合后 Q 反升
  H-c 分辨率错配 —— γ=1.0 在重连图上的最优性与观测图不同
诊断：比较 观测 / 度保持重连 / 配置模型 / 权重重排 四类图的 Q 与模块数
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


def q_of(g, lab, weights):
    return g.modularity(lab, weights=weights)


def analyse(tom, genes):
    n = len(genes)
    g = GA.to_ig(tom, genes)
    w = np.array(g.es["weight"], dtype=float)
    edges = [e.tuple for e in g.es]
    res = {}

    def leiden_q(gr, ww):
        p = la.find_partition(gr, la.RBConfigurationVertexPartition,
                              weights=ww, resolution_parameter=1.0, seed=SEED)
        return gr.modularity(p, weights=ww), len(p)

    # 1) 观测
    res["observed"] = leiden_q(g, w)

    # 2) 度保持重连（igraph C 实现），权重从观测分布重采样
    rng = np.random.default_rng(SEED)
    qs, ms = [], []
    for _ in range(10):
        gr = g.copy()
        gr.rewire(n=3 * gr.ecount(), mode="simple")
        ww = rng.choice(w, size=gr.ecount(), replace=True) if len(w) else np.ones(gr.ecount())
        q, m = leiden_q(gr, ww)
        qs.append(q); ms.append(m)
    res["rewired"] = (float(np.mean(qs)), float(np.mean(ms)))

    # 3) 权重重排（保持拓扑，打散权重-结构耦合）
    qs, ms = [], []
    for _ in range(10):
        ww = rng.permutation(w)
        q, m = leiden_q(g, ww)
        qs.append(q); ms.append(m)
    res["permuted_w"] = (float(np.mean(qs)), float(np.mean(ms)))

    # 4) 均匀权重（忽略权重信息）
    q, m = leiden_q(g, np.ones(len(w)))
    res["uniform_w"] = (float(q), float(m))

    # 诊断指标
    deg = np.array(g.degree())
    res["deg_cv"] = float(deg.std() / (deg.mean() + 1e-9))
    res["w_autocorr"] = float(np.corrcoef(
        w, [g.strength([e.source for e in g.es], weights=None)[0]] * len(w))[0, 1]) \
        if len(w) > 3 else np.nan
    # 权重与"边两端度的几何均值"的相关（权重-结构耦合强度）
    dgm = np.array([np.sqrt(deg[e[0]] * deg[e[1]]) for e in edges], dtype=float)
    if len(w) > 3 and dgm.std() > 0:
        res["w_vs_degcorr"] = float(np.corrcoef(w, dgm)[0, 1])
    else:
        res["w_vs_degcorr"] = np.nan
    return res


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
            genes = list(genes)
            tom = GA.compute_tom(df.loc[genes])
            r = analyse(tom, genes)
            r.update({"gse": gse, "pathway": pw[:30], "n_genes": len(genes)})
            rows.append(r)
            print(f"  {gse} {pw[:26]}: obs={r['observed'][0]:.3f} "
                  f"rewire={r['rewired'][0]:.3f} permW={r['permuted_w'][0]:.3f} "
                  f"unifW={r['uniform_w'][0]:.3f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "null_diagnosis.csv", index=False)
    print(f"\n共 {len(d)} 组合")
    print("\n=== 各类图的模块度（均值）===")
    for c in ["observed", "rewired", "permuted_w", "uniform_w"]:
        vals = [x[0] for x in d[c]]
        mods = [x[1] for x in d[c]]
        print(f"  {c:12s} Q={np.mean(vals):.4f}  模块数={np.mean(mods):.1f}")
    print("\n=== 诊断 ===")
    print(f"  度异质性 CV        {d.deg_cv.mean():.3f}")
    print(f"  权重 vs 度乘积相关  {d.w_vs_degcorr.mean():+.3f}")
    dz = (np.array([x[0] for x in d.observed]) -
          np.array([x[0] for x in d.rewired])) / (np.std([x[0] for x in d.rewired]) + 1e-9)
    print(f"  观测-重连 差值      {np.mean([x[0] for x in d.observed]) - np.mean([x[0] for x in d.rewired]):+.4f}")


if __name__ == "__main__":
    main()
