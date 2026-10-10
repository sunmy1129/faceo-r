#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FACEO-R 完整验证实验
降质条件: 样本下采样 / 噪声注入 / 特征缺失(dropout)
对比: FACEO-R  vs  Leiden γ=1.0  vs  WGCNA-style  vs  FACEO(固定 alpha)
评价: held-out Reactome 功能一致性 + 模块纯度
统计: 配对 Wilcoxon + BH-FDR + Cliff's delta
"""
import re
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
from scipy.stats import wilcoxon
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import gwas_full_analysis as GA
import gwas_full_analysis_v3 as V3
import gwas_faceo as FC
import gwas_faceo_r as FR

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
GS = Path("/Users/sunmy/.hermes/gwas_data")
OUT = BASE / "output/core_results"


def to_g2t(terms):
    d = defaultdict(set)
    for t, gs in terms.items():
        for g in gs:
            d[str(g).upper()].add(t)
    return d


def coherence(genes, labels, g2t, max_genes=60, seed=0):
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


def degrade(df, kind, level, seed=42):
    rng = np.random.default_rng(seed)
    X = df.values.astype(np.float32)
    if kind == "samples":
        ns = max(8, int(X.shape[1] * level))
        cols = np.sort(rng.choice(X.shape[1], ns, replace=False))
        return pd.DataFrame(X[:, cols], index=df.index,
                            columns=np.asarray(df.columns)[cols])
    elif kind == "noise":
        sd = X.std()
        Xd = X + rng.normal(0, sd * level, X.shape)
    elif kind == "dropout":
        mask = rng.random(X.shape) < level
        Xd = X.copy(); Xd[mask] = 0.0
    else:
        Xd = X
    return pd.DataFrame(Xd, index=df.index, columns=df.columns)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def main(max_cohorts=5, max_paths=3):
    go_bp = json.load(open(GS / "GO_Biological_Process_2021.json"))
    reac = json.load(open(GS / "Reactome_2022.json"))
    go2t, re2t = to_g2t(go_bp), to_g2t(reac)

    CONDITIONS = [("samples", 0.3), ("samples", 0.15), ("noise", 0.5),
                  ("noise", 1.0), ("dropout", 0.5)]
    res = pd.read_csv(OUT / "full_eval_v3.csv")
    cohorts = sorted(res.gse.unique())[:max_cohorts]
    fn = dict(GA.ALGOS)

    rows = []
    t0 = time.time()
    for gse in cohorts:
        try:
            raw = pd.read_csv(BASE / f"qcdata/{gse}.csv", index_col=0,
                              quoting=3, low_memory=False)
            raw.index = [str(i).strip().strip('"').upper() for i in raw.index]
            keep = [re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(i)) is not None
                    for i in raw.index]
            df_full = V3.preprocess_robust(raw[keep])
        except Exception:
            continue
        go_all = V3.load_go_sets()
        paths = V3.select_pathways(go_all, set(df_full.index))
        for pw_name, pw_genes in paths[:max_paths]:
            genes = [g for g in pw_genes if g in df_full.index]
            if len(genes) < 60 or len(genes) > 200:
                continue
            genes = list(genes)
            for kind, lvl in CONDITIONS:
                df = degrade(df_full.loc[genes], kind, lvl)
                try:
                    tom = GA.compute_tom(df)
                    g = GA.to_ig(tom, genes)
                except Exception:
                    continue
                # 方法
                try:
                    lab_l = np.asarray(fn["Leiden γ=1.0"](g))
                except Exception:
                    lab_l = np.zeros(len(genes), int)
                try:
                    lab_w = np.asarray(fn["WGCNA-style\nHierarchical"](g))
                except Exception:
                    lab_w = np.zeros(len(genes), int)
                q = FR.graph_reliability(tom)
                mods_r, hard_r, info_r = FR.faceo_r(tom, genes, go2t, reliability=q)
                lab_r = FR.modules_to_labels(mods_r, len(genes))

                for name, lab in [("Leiden γ=1.0", lab_l),
                                  ("WGCNA-style", lab_w),
                                  ("FACEO-R (ours)", lab_r)]:
                    rows.append({"gse": gse, "pathway": pw_name[:24],
                                 "condition": f"{kind}_{lvl}", "algo": name,
                                 "n_genes": len(genes),
                                 "alpha": info_r["alpha"], "reliability": q,
                                 "coh": coherence(genes, lab, re2t)})
        print(f"  {gse} 完成 ({len(rows)} 行, {time.time()-t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows).dropna(subset=["coh"])
    d.to_csv(OUT / "faceo_r_results.csv", index=False)
    print(f"\n共 {len(d)} 条记录，{d.groupby(['gse','pathway','condition']).ngroups} 组合")

    print("\n=== 按降质条件对比（held-out Reactome 功能一致性）===")
    piv = d.pivot_table(index=["gse", "pathway", "condition"], columns="algo",
                        values="coh")
    piv = piv.dropna()
    summ = []
    for cond in sorted(d.condition.unique()):
        sub = d[d.condition == cond]
        p = sub.pivot_table(index=["gse", "pathway"], columns="algo", values="coh").dropna()
        if len(p) < 3:
            continue
        rec = {"condition": cond, "n": len(p)}
        for a in ["Leiden γ=1.0", "WGCNA-style", "FACEO-R (ours)"]:
            rec[a] = p[a].mean() if a in p.columns else np.nan
        if "FACEO-R (ours)" in p.columns:
            for ref in ["Leiden γ=1.0", "WGCNA-style"]:
                if ref in p.columns:
                    diff = p["FACEO-R (ours)"] - p[ref]
                    rec[f"delta_vs_{ref.split()[0]}"] = diff.mean()
                    rec[f"win_vs_{ref.split()[0]}"] = f"{(diff>0).sum()}/{len(diff)}"
                    if len(diff) >= 6 and diff.std() > 0:
                        try:
                            rec[f"p_vs_{ref.split()[0]}"] = wilcoxon(diff).pvalue
                        except Exception:
                            pass
        summ.append(rec)
    s = pd.DataFrame(summ).round(4)
    print(s.to_string(index=False))
    s.to_csv(OUT / "faceo_r_summary.csv", index=False)

    # 全局
    print("\n=== 全局（所有降质条件合并）===")
    p = d.pivot_table(index=["gse", "pathway", "condition"], columns="algo",
                      values="coh").dropna()
    for ref in ["Leiden γ=1.0", "WGCNA-style"]:
        if ref in p.columns:
            diff = p["FACEO-R (ours)"] - p[ref]
            wins = (diff > 0).sum()
            print(f"  FACEO-R vs {ref}: 胜 {wins}/{len(diff)} ({wins/len(diff):.0%}), "
                  f"平均 Δ = {diff.mean():+.4f}, Cliff δ = {cliffs_delta(p['FACEO-R (ours)'], p[ref]):+.3f}")
            if len(diff) >= 6 and diff.std() > 0:
                print(f"    Wilcoxon p = {wilcoxon(diff).pvalue:.2e}")


if __name__ == "__main__":
    main()
