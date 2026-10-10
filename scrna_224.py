#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
import scanpy as sc
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import gwas_full_analysis as GA
import gwas_faceo_r as FR

D = Path.home() / ".hermes/gwas_data/scrna2"
GS = Path.home() / ".hermes/gwas_data"
OUT = Path("/Users/sunmy/Documents/QF/GWAS/output/core_results")
N_CELLS = 6000


def load_donors():
    ads = []
    for d in ["Donor1", "Donor2", "Donor3"]:
        cand_files = [
            D / f"GSE185224_GSE185224_{d}_filtered_feature_bc_matrix.h5",
            D / f"GSE185224_{d}_filtered_feature_bc_matrix.h5",
        ]
        f = next((c for c in cand_files if c.exists()), None)
        if f is None:
            print(f"  缺 {d}")
            continue
        try:
            a = sc.read_10x_h5(str(f))
        except Exception as e:
            print(f"  {d} 读取失败(可能未下完): {type(e).__name__}")
            continue
        a.var_names_make_unique()
        a.obs["donor"] = d
        ads.append(a)
        print(f"  {d}: {a.shape[0]} 细胞 x {a.shape[1]} 基因")
    if not ads:
        return None
    ad = ads[0] if len(ads) == 1 else ads[0].concatenate(*ads[1:])
    return ad


def preprocess(ad):
    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.filter_genes(ad, min_cells=3)
    ad.obs["ng"] = (ad.X > 0).sum(1).A1 if hasattr(ad.X, "A1") else (ad.X > 0).sum(1)
    ad.obs["mtp"] = np.asarray(
        ad[:, [g for g in ad.var_names if g.startswith("MT-")]].X.sum(1)
    ).ravel() / np.maximum(ad.obs["ng"].values, 1)
    ad = ad[(ad.obs["ng"] >= 300) & (ad.obs["mtp"] < 0.25)].copy()
    return ad


def _coh(genes, labels, ref2t, maxg=60, seed=0):
    vals = []
    for m in set(labels):
        mod = [genes[i] for i in range(len(genes)) if labels[i] == m]
        if len(mod) < 3:
            continue
        sets = [ref2t.get(g.upper()) for g in mod]
        sets = [s for s in sets if s and len(s) >= 3]
        if len(sets) < 3:
            continue
        if len(sets) > maxg:
            idx = np.random.default_rng(seed).choice(len(sets), maxg, replace=False)
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


def main():
    t0 = time.time()
    go_bp = json.load(open(GS / "GO_Biological_Process_2021.json"))
    print(f"GO-BP: {len(go_bp)}")
    ad = load_donors()
    if ad is None:
        print("无数据")
        return
    ad = preprocess(ad)
    print(f"QC 后: {ad.shape[0]} 细胞")
    n = min(N_CELLS, ad.shape[0])
    sc.pp.subsample(ad, n_obs=n, random_state=0)
    X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
    Xg = pd.DataFrame(X.T, index=ad.var_names, columns=ad.obs_names)
    print(f"分析矩阵: {Xg.shape[0]} 基因 x {Xg.shape[1]} 细胞")

    go_all = {k: set(str(g).upper() for g in v) for k, v in go_bp.items()}
    reac = json.load(open(GS / "Reactome_2022.json"))
    ref2t = {}
    for k, v in reac.items():
        for g in v:
            ref2t.setdefault(str(g).upper(), set()).add(k)
    avail = set(Xg.index)
    PREFER = ["apoptotic", "inflammatory response", "immune response",
              "cytokine", "interferon", "cell cycle", "oxidative stress"]
    cand = []
    for nm, s in go_all.items():
        gs = [g for g in s if g in avail]
        if 40 <= len(gs) <= 300:
            cand.append((nm, gs))
    pref = [c for c in cand if any(p.lower() in c[0].lower() for p in PREFER)]
    paths = (pref + [c for c in cand if c not in pref])[:20]
    print(f"通路: {len(paths)} (优先 {len(pref)})")

    rows = []
    for pw, genes in paths:
        sub = Xg.loc[genes].values
        C = np.corrcoef(sub)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        C = (C + C.T) / 2.0
        tom = np.abs(C)
        np.fill_diagonal(tom, 0.0)
        g = GA.to_ig(tom, genes)
        for aname in ["Leiden γ=1.0", "WGCNA-style\nHierarchical"]:
            lab = np.asarray(GA.ALGOS[aname](g))
            rows.append({"pathway": pw, "algo": aname.split("\n")[0],
                         "mods": len(set(lab)), "n": len(genes),
                         "coh": _coh(genes, lab, ref2t)})
        q = FR.graph_reliability(tom)
        mods, hard, info = FR.faceo_r(tom, genes, go_all, reliability=q)
        hard = np.asarray(hard)
        rows.append({"pathway": pw, "algo": "FACEO-R (ours)",
                     "mods": len(set(hard)), "n": len(genes),
                     "coh": _coh(genes, hard, ref2t)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "scrna_185224.csv", index=False)
    print(d.groupby("algo").agg({"coh": ["mean", "count"], "mods": "mean"}).round(4))
    print(f"完成 {time.time()-t0:.0f}s -> scrna_185224.csv")


if __name__ == "__main__":
    main()
