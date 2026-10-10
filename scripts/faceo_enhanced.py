#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FACEO-R 增强实验：
  (1) 增加 clean 条件（无降质对照）—— 回答"全场景 vs 限定场景"
  (2) α 消融：固定 α=0.35 / 0.65 / 0.95 vs 自适应 α(q)
  (3) 全部 16 cohort（扩大规模）
  (4) 多核并行
"""
import sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import gwas_full_analysis as GA
import gwas_full_analysis_v3 as V3
import gwas_faceo_r as FR

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
GS = Path("/Users/sunmy/.hermes/gwas_data")
OUT = BASE / "output/core_results"
NPROC = 10

CONDITIONS = [("clean", 0.0), ("samples", 0.3), ("samples", 0.15),
              ("noise", 0.5), ("noise", 1.0), ("dropout", 0.5)]


def to_g2t(terms):
    from collections import defaultdict
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


_G = {}


def _init():
    go_bp = json.load(open(GS / "GO_Biological_Process_2021.json"))
    reac = json.load(open(GS / "Reactome_2022.json"))
    _G["go2t"] = to_g2t(go_bp)
    _G["re2t"] = to_g2t(reac)
    _G["fn"] = dict(GA.ALGOS)


def _work(job):
    gse, pw_name, pw_genes, kind, lvl = job
    out = []
    try:
        raw = pd.read_csv(BASE / f"qcdata/{gse}.csv", index_col=0,
                          quoting=3, low_memory=False)
        raw.index = [str(i).strip().strip('"').upper() for i in raw.index]
        import re
        keep = [re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(i)) is not None
                for i in raw.index]
        df_full = V3.preprocess_robust(raw[keep])
    except Exception:
        return out
    genes = [g for g in pw_genes if g in df_full.index]
    if len(genes) < 60 or len(genes) > 200:
        return out
    genes = list(genes)
    df = degrade(df_full.loc[genes], kind, lvl)
    try:
        tom = GA.compute_tom(df)
        g = GA.to_ig(tom, genes)
    except Exception:
        return out
    fn = _G["fn"]
    try:
        lab_l = np.asarray(fn["Leiden γ=1.0"](g))
    except Exception:
        lab_l = np.zeros(len(genes), int)
    try:
        lab_w = np.asarray(fn["WGCNA-style\nHierarchical"](g))
    except Exception:
        lab_w = np.zeros(len(genes), int)
    q = FR.graph_reliability(tom)
    runs = [("FACEO-R (ours)", None)]
    runs += [(f"alpha={a}", a) for a in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55)]
    runs += [("FACEO-R-narrow", ("narrow", 0.05, 0.25))]
    got = {"Leiden γ=1.0": lab_l, "WGCNA-style": lab_w}
    for nm, fixed in runs:
        try:
            if fixed is None:
                mods, hard, info = FR.faceo_r(tom, genes, _G["go2t"], reliability=q)
            elif isinstance(fixed, tuple):
                mods, hard, info = FR.faceo_r(tom, genes, _G["go2t"],
                                              alpha_min=fixed[1], alpha_max=fixed[2],
                                              reliability=q)
            else:
                mods, hard, info = FR.faceo_r(tom, genes, _G["go2t"],
                                              alpha_min=fixed, alpha_max=fixed,
                                              reliability=q)
            got[nm] = FR.modules_to_labels(mods, len(genes))
            a_val = info["alpha"]
        except Exception:
            got[nm] = np.zeros(len(genes), int); a_val = np.nan
        out.append({"gse": gse, "pathway": pw_name[:24], "condition": f"{kind}_{lvl}",
                    "algo": nm, "n_genes": len(genes), "alpha": a_val,
                    "reliability": q, "coh": coherence(genes, got[nm], _G["re2t"])})
    for nm in ["Leiden γ=1.0", "WGCNA-style"]:
        out.append({"gse": gse, "pathway": pw_name[:24], "condition": f"{kind}_{lvl}",
                    "algo": nm, "n_genes": len(genes), "alpha": np.nan,
                    "reliability": q, "coh": coherence(genes, got[nm], _G["re2t"])})
    return out


def main():
    res = pd.read_csv(OUT / "full_eval_v3.csv")
    cohorts = sorted(res.gse.unique())
    go_all = V3.load_go_sets()
    jobs = []
    for gse in cohorts:
        try:
            raw = pd.read_csv(BASE / f"qcdata/{gse}.csv", index_col=0,
                              quoting=3, low_memory=False)
            allg = set(str(i).strip().strip('"').upper() for i in raw.index)
        except Exception:
            continue
        paths = V3.select_pathways(go_all, allg)
        for pn, pg in paths[:3]:
            for k, l in CONDITIONS:
                jobs.append((gse, pn, list(pg), k, l))
    print(f"任务数: {len(jobs)} | 队列: {len(cohorts)} | 进程: {NPROC}", flush=True)
    t0 = time.time()
    rows = []
    with Pool(NPROC, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=2), 1):
            rows += r
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows).dropna(subset=["coh"])
    d.to_csv(OUT / "faceo_r_enhanced.csv", index=False)
    print(f"\n共 {len(d)} 条 | {d.groupby(['gse','pathway','condition']).ngroups} 组合 | {time.time()-t0:.0f}s")
    return d


if __name__ == "__main__":
    d = main()
