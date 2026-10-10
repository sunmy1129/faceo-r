#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
假设验证：样本量下降时，Leiden 退化有多快？功能先验能否补偿？
若 Leiden 在降质图上崩而功能先验稳定 -> 「鲁棒性路线」成立
"""
import re
import sys
import json
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import gwas_full_analysis as GA
import gwas_full_analysis_v3 as V3
import gwas_faceo as FC

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
GS = Path("/Users/sunmy/.hermes/gwas_data")
SUBSETS = [1.0, 0.5, 0.3, 0.2]


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


def main(max_cohorts=4, max_paths=3):
    go_bp = json.load(open(GS / "GO_Biological_Process_2021.json"))
    reac = json.load(open(GS / "Reactome_2022.json"))
    go2t, re2t = to_g2t(go_bp), to_g2t(reac)
    print("验证：样本降质下 Leiden vs 功能先验增强")

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
            n_samp = df_full.shape[1]
            for frac in SUBSETS:
                ns = max(8, int(n_samp * frac))
                rng = np.random.default_rng(42)
                cols = rng.choice(n_samp, ns, replace=False)
                df = df_full.iloc[:, sorted(cols)]
                tom = GA.compute_tom(df.loc[genes])
                g = GA.to_ig(tom, genes)
                # Leiden（仅表达）
                lab_l = np.asarray(GA.ALGOS["Leiden γ=1.0"](g))
                # FACEO（功能先验增强）
                mods_f, hard_f, _ = FC.faceo(tom, genes, go2t, alpha=0.6,
                                             gamma=1.0, tau=0.65)
                lab_f = np.array([0] * len(genes))
                for mi, m in enumerate(mods_f):
                    for i in m:
                        lab_f[i] = mi
                rows.append({"gse": gse, "pathway": pw_name[:24], "frac": frac,
                             "n_samples": ns,
                             "c_leiden": coherence(genes, lab_l, re2t),
                             "c_faceo": coherence(genes, lab_f, re2t)})
        print(f"  {gse} 完成", flush=True)

    d = pd.DataFrame(rows).dropna()
    d.to_csv(BASE / "output/core_results/degradation_test.csv", index=False)
    print(f"\n共 {len(d)} 条")
    agg = d.groupby("frac").agg(
        leiden=("c_leiden", "mean"), faceo=("c_faceo", "mean"),
        n=("c_leiden", "count")).round(4)
    agg["delta"] = (agg.faceo - agg.leiden).round(4)
    print("\n=== 按样本保留比例对比（held-out Reactome）===")
    print(agg.to_string())
    print("\n判定:")
    full = agg.loc[1.0, "delta"] if 1.0 in agg.index else np.nan
    low = agg.loc[agg.index.min(), "delta"]
    print(f"  完整数据 delta = {full:+.4f}")
    print(f"  最低样本 delta = {low:+.4f}")
    print(f"  {'✅ 降质时功能先验优势扩大，鲁棒性路线成立' if low > full else '❌ 优势未扩大'}")


if __name__ == "__main__":
    main()
