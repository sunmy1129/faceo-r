#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并 v3 分数据集结果 -> 统一长表 + 汇总统计"""
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
OUT = Path("/Users/sunmy/Documents/QF/GWAS/output/core_results")


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    r = p[o]
    adj = r * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(adj, 0, 1)
    return out


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    n = len(a) * len(b)
    return (gt - lt) / n if n else np.nan


def merge():
    fs = [f for f in glob.glob(str(OUT / "v3_*.csv"))
          if "GSE10072_GSE137342" not in f]
    dfs = []
    for f in fs:
        try:
            d = pd.read_csv(f)
            if len(d):
                dfs.append(d)
        except Exception:
            continue
    if not dfs:
        print("无结果文件")
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["gse", "pathway", "algo"], keep="last")
    df["algo"] = df["algo"].astype(str)
    df.to_csv(OUT / "full_eval_v3.csv", index=False)
    print(f"合并 {len(fs)} 个文件 -> {len(df)} 条记录")
    print(f"数据集: {df.gse.nunique()} | 通路: {df.pathway.nunique()} | 算法: {df.algo.nunique()}")
    print(f"组合数: {df.groupby(['gse','pathway']).ngroups}")
    return df


def summarize(df):
    cols = ["n_modules", "modularity_Q", "coh_z_mean", "stability_ARI",
            "antag_split_rate", "ARI_noise0.1"]
    cols = [c for c in cols if c in df.columns]
    g = df.groupby("algo")[cols].mean().round(3)
    g["n_obs"] = df.groupby("algo").size()
    g = g.sort_values("coh_z_mean", ascending=False)
    print("\n" + "=" * 100)
    print("各算法汇总（跨全部数据集×通路）")
    print("=" * 100)
    print(g)
    g.to_csv(OUT / "algo_summary_v3.csv")
    return g


def paired_tests(df, ref="WGCNA-style Hierarchical", key="Leiden γ=1.0"):
    rows = []
    for metric, desc in [("n_modules", "模块数"), ("modularity_Q", "模块度Q"),
                         ("coh_z_mean", "功能内聚z"), ("stability_ARI", "稳定性"),
                         ("antag_split_rate", "拮抗拆分率")]:
        if metric not in df.columns:
            continue
        sub = df[df.algo.isin([ref, key])][["gse", "pathway", "algo", metric]].dropna()
        piv = sub.pivot_table(index=["gse", "pathway"], columns="algo", values=metric)
        if ref not in piv.columns or key not in piv.columns:
            continue
        x, y = piv[key].values, piv[ref].values
        if len(x) < 5 or np.allclose(x, y):
            continue
        try:
            _, p = stats.wilcoxon(x, y)
        except Exception:
            continue
        rows.append({"metric": metric, "description": desc, "n_pairs": len(x),
                     "median_experimental": np.median(x), "median_reference": np.median(y),
                     "p_value": p, "cliffs_delta": cliffs(x, y)})
    if not rows:
        return None
    r = pd.DataFrame(rows)
    r["p_fdr"] = bh_fdr(r["p_value"])
    r["significant"] = r["p_fdr"] < 0.05
    r.to_csv(OUT / "statistics_v3.csv", index=False)
    print("\n" + "=" * 100)
    print(f"配对检验（{key} vs {ref}）")
    print("=" * 100)
    for _, x in r.iterrows():
        sig = "✓显著" if x["significant"] else "✗"
        print(f"  {x['description']:<12} n={int(x['n_pairs']):>3}  "
              f"{x['median_experimental']:>8.3f} vs {x['median_reference']:>8.3f}  "
              f"p_FDR={x['p_fdr']:.2e} {sig}  δ={x['cliffs_delta']:+.3f}")
    return r


if __name__ == "__main__":
    df = merge()
    if df is not None and len(df):
        summarize(df)
        paired_tests(df)
        print(f"\n✅ 已生成 full_eval_v3.csv / algo_summary_v3.csv / statistics_v3.csv")
