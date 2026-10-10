#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决定性前置检验：GO 功能一致性 与 Reactome 功能一致性 的相关性
若两者高度正相关 -> FGO（优化 GO）必然在 Reactome 上赢
"""
import re
import sys
import json
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import gwas_full_analysis as GA
import gwas_full_analysis_v3 as V3

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
GS = Path("/Users/sunmy/.hermes/gwas_data")
GAMMAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def to_g2t(terms):
    d = defaultdict(set)
    for t, gs in terms.items():
        for g in gs:
            d[str(g).upper()].add(t)
    return d


def coherence(genes, labels, g2t, max_genes=60, seed=0):
    """模块平均的功能一致性（模块内基因对注释 Jaccard 均值）"""
    vals = []
    for m in set(labels):
        mod = [genes[i] for i in range(len(genes)) if labels[i] == m]
        if len(mod) < 3:
            continue
        sets = [g2t.get(g.upper()) for g in mod]
        sets = [s for s in sets if s and len(s) >= 3]
        if len(sets) < 3:
            continue
        if len(sets) > max_genes:
            idx = np.random.default_rng(seed).choice(len(sets), max_genes, replace=False)
            sets = [sets[i] for i in idx]
        sims = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = len(sets[i] | sets[j])
                if u:
                    sims.append(len(sets[i] & sets[j]) / u)
        if sims:
            vals.append(float(np.mean(sims)))
    return float(np.mean(vals)) if vals else np.nan


def main(max_cohorts=6):
    go_bp = json.load(open(GS / "GO_Biological_Process_2021.json"))
    reac = json.load(open(GS / "Reactome_2022.json"))
    go2t, re2t = to_g2t(go_bp), to_g2t(reac)
    print(f"GO-BP {len(go_bp)} | Reactome {len(reac)}")

    res = pd.read_csv(BASE / "output/core_results/full_eval_v3.csv")
    cohorts = sorted(res.gse.unique())[:max_cohorts]
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
        for pw_name, pw_genes in paths[:5]:
            genes = [g for g in pw_genes if g in df.index]
            if len(genes) < 60 or len(genes) > 250:
                continue
            genes = list(genes)
            tom = GA.compute_tom(df.loc[genes])
            g = GA.to_ig(tom, genes)
            for gm in GAMMAS:
                fn = GA.ALGOS.get(f"Leiden γ={gm}")
                if fn is None:
                    continue
                lab = np.asarray(fn(g))
                rows.append({"gse": gse, "pathway": pw_name[:26], "gamma": gm,
                             "c_go": coherence(genes, lab, go2t),
                             "c_re": coherence(genes, lab, re2t)})
        print(f"  {gse} 完成 ({len(rows)} 行)", flush=True)

    d = pd.DataFrame(rows).dropna()
    d.to_csv(BASE / "output/core_results/go_reactome_correlation.csv", index=False)
    print(f"\n共 {len(d)} 个 (组合,γ) 观测")
    rho, p = spearmanr(d.c_go, d.c_re)
    r, pr = pearsonr(d.c_go, d.c_re)
    print(f"\n=== GO 一致性 vs Reactome 一致性 ===")
    print(f"  Spearman ρ = {rho:+.3f}  (p = {p:.2e})")
    print(f"  Pearson  r = {r:+.3f}  (p = {pr:.2e})")
    print(f"  判定: {'✅ 高度正相关，FGO 路线可行' if rho > 0.5 else '⚠️ 相关不足，需谨慎' if rho > 0.3 else '❌ 相关弱，路线不可行'}")

    # 关键：在同一通路内，GO 最优的 γ 是否也是 Reactome 最优的 γ？
    hit = 0; tot = 0
    for (gse, pw), sub in d.groupby(["gse", "pathway"]):
        if len(sub) < 4:
            continue
        tot += 1
        if sub.loc[sub.c_go.idxmax(), "gamma"] == sub.loc[sub.c_re.idxmax(), "gamma"]:
            hit += 1
    if tot:
        print(f"\n  同一通路内，GO-最优 γ == Reactome-最优 γ 的比例: {hit}/{tot} ({hit/tot:.0%})")
        print(f"  (随机基线 = {1/len(GAMMAS):.0%})")


if __name__ == "__main__":
    main()
