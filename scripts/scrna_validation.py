#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单细胞验证：FACEO-R 在真实 scRNA-seq 数据上的鲁棒性
数据：GSE145926（COVID-19 BALF，10x Genomics）
分组：mild (n=3) / severe (n=6)

两种场景对比：
  A. scRNA（原始单细胞，真实 dropout）—— 期望 FACEO-R 优势扩大
  B. pseudobulk（按样本聚合，接近 bulk）—— 期望差距缩小

评价：held-out Reactome 功能一致性
"""
import re
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import scanpy as sc
import gwas_full_analysis as GA
import gwas_faceo_r as FR

SCR = Path("/Users/sunmy/.hermes/gwas_data/scrna")
GS = Path("/Users/sunmy/.hermes/gwas_data")
OUT = Path("/Users/sunmy/Documents/QF/GWAS/output/core_results")
N_CELLS = 2000          # 每样本降采样细胞数
SEED = 42


def to_g2t(terms):
    d = defaultdict(set)
    for t, gs in terms.items():
        for g in gs:
            d[str(g).upper()].add(t)
    return d


def load_groups():
    g = {}
    for l in (SCR / "groups.txt").read_text().strip().split("\n"):
        if "\t" in l:
            k, v = l.split("\t")
            g[k] = "severe" if "severe" in v else ("mild" if v == "mild" else "NA")
    return g


def load_scrna(max_cells=N_CELLS):
    """加载并合并单细胞数据"""
    groups = load_groups()
    adatas, labels = [], []
    for f in sorted(SCR.glob("*.h5")):
        sid = f.name.split("_")[1]
        grp = groups.get(sid, "NA")
        if grp == "NA":
            continue
        try:
            a = sc.read_10x_h5(f)
            a.var_names_make_unique()
            if a.n_obs > max_cells:
                sc.pp.subsample(a, n_obs=max_cells, random_state=SEED)
            a.obs["sample"] = sid
            a.obs["group"] = grp
            adatas.append(a)
            labels.append(grp)
        except Exception as e:
            print(f"  skip {sid}: {e}")
    ad = sc.concat(adatas, join="outer", fill_value=0)
    print(f"合并: {ad.n_obs} 细胞 x {ad.n_vars} 基因")
    return ad


def preprocess_sc(ad):
    """QC + 标准化"""
    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.filter_genes(ad, min_cells=10)
    ad.layers["counts"] = ad.X.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ad.var_names = [str(v).upper() for v in ad.var_names]
    ad.var_names_make_unique()
    print(f"QC 后: {ad.n_obs} 细胞 x {ad.n_vars} 基因")
    return ad


def pseudobulk(ad):
    """按（样本 × 细胞类型）聚合 -> pseudobulk 表达矩阵"""
    # 用 leiden 分群作为细胞类型代理
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat")
    ad_h = ad[:, ad.var.highly_variable].copy()
    sc.pp.scale(ad_h, max_value=10)
    sc.tl.pca(ad_h, n_comps=30)
    sc.pp.neighbors(ad_h, n_neighbors=15)
    sc.tl.leiden(ad_h, resolution=0.5, key_added="ct")
    ad.obs["ct"] = ad_h.obs["ct"].values
    # 聚合
    df = pd.DataFrame(ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X,
                      index=ad.obs_names, columns=ad.var_names)
    df["sample"] = ad.obs["sample"].values
    df["ct"] = ad.obs["ct"].values
    pb = df.groupby(["sample", "ct"]).mean()
    print(f"pseudobulk: {pb.shape[0]} 样本-细胞类型组合 x {pb.shape[1]} 基因")
    return pb


def main():
    t0 = time.time()
    reac = json.load(open(GS / "Reactome_2022.json"))
    go_bp = json.load(open(GS / "GO_Biological_Process_2021.json"))
    re2t, go2t = to_g2t(reac), to_g2t(go_bp)
    print(f"Reactome {len(reac)} | GO-BP {len(go_bp)}")

    ad = load_scrna()
    ad = preprocess_sc(ad)
    X = pd.DataFrame(ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X,
                     index=ad.obs_names, columns=ad.var_names)
    # 转置为 基因 x 细胞
    Xg = X.T

    go_all = {}
    for k, v in go_bp.items():
        go_all[k] = set(str(g).upper() for g in v)
    avail = set(Xg.index)
    # 优先选择论文中的经典通路，不足时按命中基因数补足
    PREFER = ["apoptotic", "inflammatory response", "immune response",
              "cytokine", "I-kappaB kinase", "interferon", "cell cycle",
              "oxidative stress", "antigen"]
    cand = []
    for n, s in go_all.items():
        gs = [g for g in s if g in avail]
        if 40 <= len(gs) <= 300:
            cand.append((n, gs))
    pref = [c for c in cand if any(p.lower() in c[0].lower() for p in PREFER)]
    rest = [c for c in cand if c not in pref]
    paths = (pref + rest)[:20]
    print(f"可用通路: {len(paths)} (优先命中 {len(pref)})")

    rows = []
    for pw, genes in paths:
        sub = Xg.loc[genes]
        # 场景 A: scRNA（细胞为样本，真实 dropout）
        try:
            C = np.corrcoef(sub.values)
            C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
            C = (C + C.T) / 2.0                      # 强制对称
            tom = np.abs(C)                          # 相关强度
            np.fill_diagonal(tom, 0.0)
            tom = tom.astype(np.float64)
            g = GA.to_ig(tom, genes)
            lab_l = np.asarray(GA.ALGOS["Leiden γ=1.0"](g))
            lab_w = np.asarray(GA.ALGOS["WGCNA-style\nHierarchical"](g))
            mods_r, hard_r, info = FR.faceo_r(tom, genes, go2t)
            lab_r = FR.modules_to_labels(mods_r, len(genes))
            dr = float((sub.values == 0).mean())
            for algo, lab in [("Leiden", lab_l), ("WGCNA", lab_w), ("FACEO-R", lab_r)]:
                rows.append({"pathway": pw[:34], "scene": "scRNA", "algo": algo,
                             "n_genes": len(genes), "n_cells": int(sub.shape[1]),
                             "dropout": dr,
                             "coh": _coh(genes, lab, re2t)})
        except Exception as e:
            print(f"  {pw[:30]}: {e}")
            continue
        print(f"  ✓ {pw[:40]} ({len(genes)} genes, dropout={rows[-1]['dropout']:.1%})", flush=True)

    d = pd.DataFrame(rows)
    if "coh" not in d.columns or len(d) == 0:
        print("⚠️ 无有效结果，检查通路/矩阵构建")
        return
    d = d.dropna(subset=["coh"])
    d.to_csv(OUT / "scrna_validation.csv", index=False)
    print(f"\n共 {len(d)} 条 ({time.time()-t0:.0f}s)")
    if len(d):
        agg = d.groupby("algo")["coh"].agg(["mean", "sem", "count"]).round(4)
        print("\n=== 单细胞验证（held-out Reactome）===")
        print(agg.to_string())
        piv = d.pivot_table(index="pathway", columns="algo", values="coh").dropna()
        if len(piv) >= 3:
            from scipy.stats import wilcoxon
            for ref in ["Leiden", "WGCNA"]:
                if ref in piv.columns:
                    diff = piv["FACEO-R"] - piv[ref]
                    w = (diff > 0).sum()
                    pv = wilcoxon(diff).pvalue if diff.std() > 0 else np.nan
                    print(f"FACEO-R vs {ref}: {w}/{len(piv)} 胜, Δ={diff.mean():+.4f}, p={pv:.2e}")


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


if __name__ == "__main__":
    main()
