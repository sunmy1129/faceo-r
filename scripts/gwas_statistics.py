#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计推断模块：配对检验 + BH-FDR 校正 + 效应量
检验 H1–H6（详见研究方案 v3.0）
输出：statistics.csv / statistics.tex
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
RES = BASE / "output/core_results/full_eval.csv"
OUT = BASE / "output/core_results"

REF = "WGCNA-style Hierarchical"   # 对照基准
KEY = "Leiden γ=1.0"               # 核心实验组（共识稳定性推荐的适中分辨率）


def cliffs_delta(a, b):
    """Cliff's delta 效应量（-1..1）"""
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    n = len(a) * len(b)
    return (gt - lt) / n if n else np.nan


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def paired(df, algo_a, algo_b, metric):
    """按 (gse, pathway) 对齐后做配对 Wilcoxon"""
    if metric not in df.columns:
        return None
    sub = df[df.algo.isin([algo_a, algo_b])][["gse", "pathway", "algo", metric]].dropna()
    piv = sub.pivot_table(index=["gse", "pathway"], columns="algo", values=metric)
    if algo_a not in piv.columns or algo_b not in piv.columns:
        return None
    x = piv[algo_a].values
    y = piv[algo_b].values
    if len(x) < 5 or np.allclose(x, y):
        return dict(n=len(x), stat=np.nan, p=np.nan, delta=np.nan,
                    median_a=float(np.median(x)) if len(x) else np.nan,
                    median_b=float(np.median(y)) if len(y) else np.nan)
    try:
        stat, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p = np.nan, np.nan
    return dict(n=len(x), stat=float(stat), p=float(p), delta=cliffs_delta(x, y),
                median_a=float(np.median(x)), median_b=float(np.median(y)))


def main():
    df = pd.read_csv(RES)
    df["algo"] = df["algo"].astype(str)
    algos = sorted(df["algo"].unique())

    rows = []
    # ---- H1–H4：核心实验组 vs 对照，四个维度 ----
    hyp_map = [
        ("H1", "n_modules", "拆分能力（模块数）"),
        ("H2", "modularity_Q", "拆分不以模块度为代价"),
        ("H3", "coh_z_mean", "功能内聚性更优"),
        ("H4", "antag_split_rate", "拮抗基因对拆分率"),
        ("H3b", "purity_mean", "富集纯度（辅助）"),
        ("H2b", "stability_ARI", "稳定性"),
        ("H2c", "ARI_noise0.1", "抗噪鲁棒性"),
    ]
    for hid, metric, desc in hyp_map:
        r = paired(df, KEY, REF, metric)
        if r:
            rows.append(dict(hypothesis=hid, metric=metric, description=desc,
                             comparison=f"{KEY} vs {REF}", **r))

    # ---- 各算法 vs 对照的功能内聚性（全表）----
    for a in algos:
        if a == REF:
            continue
        r = paired(df, a, REF, "coh_z_mean")
        if r:
            rows.append(dict(hypothesis="H3-all", metric="coh_z_mean",
                             description="功能内聚性", comparison=f"{a} vs {REF}", **r))

    res = pd.DataFrame(rows)
    res["p_fdr"] = bh_fdr(res["p"].fillna(1.0).values)
    res["significant"] = res["p_fdr"] < 0.05
    res["effect_size"] = res["delta"].apply(
        lambda d: "negligible" if d != d or abs(d) < 0.147 else
                  ("small" if abs(d) < 0.33 else ("medium" if abs(d) < 0.474 else "large")))
    res.to_csv(OUT / "statistics.csv", index=False)

    # ---- LaTeX 表 ----
    def tex(df_sub):
        lines = [r"\begin{tabular}{llrrrrl}", r"\toprule",
                 r"Hypothesis & Metric & Median (A) & Median (B) & $p_{\mathrm{FDR}}$ & $\delta$ & Effect \\",
                 r"\midrule"]
        for _, r in df_sub.iterrows():
            lines.append(
                f"{r['hypothesis']} & {r['metric']} & {r['median_a']:.3f} & {r['median_b']:.3f} & "
                f"{r['p_fdr']:.2e} & {r['delta']:.3f} & {r['effect_size']} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    tex_main = tex(res[res.hypothesis.isin(["H1", "H2", "H3", "H4", "H3b", "H2b", "H2c"])])
    (OUT / "statistics.tex").write_text(tex_main, encoding="utf-8")

    # ---- 打印摘要 ----
    print("="*100)
    print(f"配对检验（实验组 {KEY}  vs  对照 {REF}，检验单位 = 数据集×通路）")
    print("="*100)
    main_tab = res[res.hypothesis.isin(["H1", "H2", "H3", "H4", "H3b", "H2b", "H2c"])]
    for _, r in main_tab.iterrows():
        sig = "✓" if r["significant"] else "✗"
        print(f"{r['hypothesis']:>4} {r['description']:<22} n={int(r['n']):>2}  "
              f"中位 {r['median_a']:>8.3f} vs {r['median_b']:>8.3f}  "
              f"p_FDR={r['p_fdr']:.2e} {sig}  δ={r['delta']:+.3f} ({r['effect_size']})")
    print()
    print(f"已保存: {OUT/'statistics.csv'}  与  {OUT/'statistics.tex'}")
    return res


if __name__ == "__main__":
    main()
