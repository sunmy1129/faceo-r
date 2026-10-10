#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终路径测试：用 GO 注释选 γ，用 held-out 的 Reactome 注释评价
（GO 与 Reactome 是独立注释体系，构成正当的 held-out 验证）
"""
import re
import sys
import gzip
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import gwas_full_analysis as GA
import gwas_full_analysis_v3 as V3

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
GS = Path("/Users/sunmy/.hermes/gwas_data")
RES = BASE / "output/core_results/full_eval_v3.csv"
GAMMAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def load_lib(fname):
    f = GS / fname
    if not f.exists():
        return {}
    return {k: set(str(x).upper() for x in v)
            for k, v in __import__("json").load(open(f)).items()}


def coherence(module_genes, g2t, max_genes=60, seed=0):
    sets = [g2t.get(g.upper()) for g in module_genes]
    sets = [s for s in sets if s and len(s) >= 3]
    if len(sets) < 3:
        return np.nan
    if len(sets) > max_genes:
        idx = np.random.default_rng(seed).choice(len(sets), max_genes, replace=False)
        sets = [sets[i] for i in idx]
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = len(sets[i] | sets[j])
            if u:
                sims.append(len(sets[i] & sets[j]) / u)
    return float(np.mean(sims)) if sims else np.nan


def coherence_z(genes, labels, g2t, n_perm=15, seed=42):
    """模块平均的标准化功能一致性"""
    zs = []
    pool_sets = [g2t.get(g.upper()) for g in genes]
    pool_sets = [s for s in pool_sets if s and len(s) >= 3]
    if len(pool_sets) < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    for m in set(labels):
        mod = [genes[i] for i in range(len(genes)) if labels[i] == m]
        if len(mod) < 3:
            continue
        obs = coherence(mod, g2t)
        if obs != obs:
            continue
        mm = min(len(mod), 60)
        null = []
        for _ in range(n_perm):
            sel = rng.choice(len(pool_sets), mm, replace=False)
            # 用注释集合直接算（避免二次查表）
            ss = [pool_sets[i] for i in sel]
            sims = []
            for a in range(len(ss)):
                for b in range(a + 1, len(ss)):
                    u = len(ss[a] | ss[b])
                    if u:
                        sims.append(len(ss[a] & ss[b]) / u)
            if sims:
                null.append(float(np.mean(sims)))
        if len(null) >= 5:
            null = np.asarray(null)
            zs.append((obs - null.mean()) / (null.std() + 1e-9))
    return float(np.mean(zs)) if zs else np.nan


def to_g2t(terms):
    from collections import defaultdict
    d = defaultdict(set)
    for t, gs in terms.items():
        for g in gs:
            d[g.upper()].add(t)
    return dict(d)


def main():
    go_bp = {}
    for fn in ["GO_Biological_Process_2021.json"]:
        go_bp.update(load_lib(fn))
    reactome = load_lib("Reactome_2022.json")
    print(f"GO BP: {len(go_bp)} terms | Reactome: {len(reactome)} terms")
    if not reactome:
        print("Reactome 未找到，退出")
        return
    go2t, re2t = to_g2t(go_bp), to_g2t(reactome)

    res = pd.read_csv(RES)
    res["algo"] = res["algo"].astype(str)
    cohorts = sorted(res.gse.unique())[:5]
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
        for pw_name, pw_genes in paths[:4]:
            genes = [g for g in pw_genes if g in df.index]
            if len(genes) < 40 or len(genes) > 200:
                continue
            genes = list(genes)
            tom = GA.compute_tom(df.loc[genes])
            g = GA.to_ig(tom, genes)
            z_go, z_re = {}, {}
            for gm in GAMMAS:
                fn = GA.ALGOS.get(f"Leiden γ={gm}")
                if fn is None:
                    continue
                lab = np.asarray(fn(g))
                z_go[gm] = coherence_z(genes, lab, go2t)
                z_re[gm] = coherence_z(genes, lab, re2t)
            if not z_go or not z_re:
                continue
            g_sel = max(z_go, key=lambda k: (z_go[k] if z_go[k] == z_go[k] else -9))
            g_oracle = max(z_re, key=lambda k: (z_re[k] if z_re[k] == z_re[k] else -9))
            rows.append({"gse": gse, "pathway": pw_name[:28],
                         "gamma_go": g_sel, "gamma_oracle": g_oracle,
                         "zre_go_selected": z_re.get(g_sel, np.nan),
                         "zre_fixed_1.0": z_re.get(1.0, np.nan),
                         "zre_oracle": z_re.get(g_oracle, np.nan)})
    if not rows:
        print("无可用组合")
        return
    d = pd.DataFrame(rows)
    d["gain_vs_fixed"] = d["zre_go_selected"] - d["zre_fixed_1.0"]
    d["gain_vs_oracle"] = d["zre_go_selected"] - d["zre_oracle"]
    print("\n" + d.to_string(index=False))
    print(f"\n=== 结果 ===")
    print(f"组合数: {len(d)}")
    print(f"GO 选 γ 在 Reactome 上优于固定 γ=1.0: {(d.gain_vs_fixed > 0).sum()}/{len(d)}")
    print(f"平均增益 vs 固定 γ=1.0: {d.gain_vs_fixed.mean():+.3f}")
    print(f"平均增益 vs oracle:     {d.gain_vs_oracle.mean():+.3f}")
    print(f"GO 选的 γ 命中 oracle 的比例: {(d.gamma_go == d.gamma_oracle).mean()*100:.0f}%")
    d.to_csv(BASE / "output/core_results/faced_holdout.csv", index=False)


if __name__ == "__main__":
    main()
