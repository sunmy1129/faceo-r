#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 5-7 (期刊级重绘): 合成基准 / 零模型 / 下游预测"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import journal_style as JS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
OUT = BASE / "output/core_results"
FIG = BASE / "output/figures_v3"
JS.apply_style()
SEL = ["Leiden γ=1.0", "WGCNA-style Hierarchical", "CPM γ=0.05", "Louvain",
       "Spectral clustering", "K-means"]


def fig5(syn):
    fig = plt.figure(figsize=(JS.DOUBLE, 2.45))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.062, right=0.99,
                          top=0.855, bottom=0.175)
    cols = [JS.C_LEIDEN, JS.C_WGCNA, JS.C_CPM, JS.NPG[4], JS.NPG[5], JS.NPG[9]]

    # (a) LFR 梯度
    ax = fig.add_subplot(gs[0, 0])
    lfr = syn[syn.benchmark == "LFR"]
    lfr = lfr[lfr.algo.astype(str).isin(SEL)]
    for algo, c in zip(SEL, cols):
        s = lfr[lfr.algo == algo].groupby("mu")["ARI"].mean()
        if len(s):
            ax.plot(s.index.astype(float), s.values, marker="o", ms=2.4, lw=0.8,
                    color=c, markeredgecolor="white", markeredgewidth=0.3,
                    label=JS.short_algo(algo))
    ax.set_xlabel("Community fuzziness $\\mu$", fontsize=6)
    ax.set_ylabel("ARI (absolute accuracy)", fontsize=6)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Block-model benchmark", fontsize=6.5, pad=3)
    JS.tidy_axis(ax, grid_axis="y"); JS.panel_label(ax, "a")
    ax.legend(fontsize=4.5, loc="lower left", ncol=1)

    # (b) 嵌套子群
    ax = fig.add_subplot(gs[0, 1])
    nes = syn[syn.benchmark == "Nested_sub"]
    nes = nes[nes.algo.astype(str).isin(SEL)].groupby("algo")["ARI"].mean()
    nes = nes.reindex([s for s in SEL if s in nes.index]).dropna()
    y = np.arange(len(nes))[::-1]
    cs = [JS.C_LEIDEN if n == "Leiden γ=1.0" else
          (JS.C_WGCNA if "WGCNA" in n else "#B0BEC5") for n in nes.index]
    ax.barh(y, nes.values, color=cs, height=0.62, edgecolor="white",
            linewidth=0.4, zorder=3)
    for yi, v in zip(y, nes.values):
        ax.text(v + 0.012, yi, f"{v:.3f}", fontsize=4.7, va="center",
                color="#444444")
    ax.set_yticks(y)
    ax.set_yticklabels([JS.short_algo(n) for n in nes.index], fontsize=4.7)
    ax.set_xlabel("ARI (recovery of 12 sub-groups)", fontsize=6)
    ax.set_xlim(0, 1.12)
    ax.set_title("Nested sub-structure", fontsize=6.5, pad=3)
    JS.tidy_axis(ax, grid_axis="x"); JS.panel_label(ax, "b", x=-0.56)

    # (c) 噪声鲁棒性
    ax = fig.add_subplot(gs[0, 2])
    nz = syn[syn.benchmark == "Noise"]
    nz = nz[nz.algo.astype(str).isin(SEL)]
    for algo, c in zip(SEL, cols):
        s = nz[nz.algo == algo].groupby("noise")["ARI"].mean()
        if len(s):
            ax.plot(s.index.astype(float), s.values, marker="o", ms=2.4, lw=0.8,
                    color=c, markeredgecolor="white", markeredgewidth=0.3)
    ax.set_xlabel("Weight noise $\\sigma$", fontsize=6)
    ax.set_ylabel("ARI", fontsize=6)
    ax.set_title("Noise robustness", fontsize=6.5, pad=3)
    JS.tidy_axis(ax, grid_axis="y"); JS.panel_label(ax, "c")

    o = FIG / "Fig5_synthetic.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")


def fig6(null):
    fig = plt.figure(figsize=(JS.ONE_HALF, 2.4))
    gs = fig.add_gridspec(1, 2, wspace=0.44, left=0.125, right=0.985,
                          top=0.855, bottom=0.175)

    ax = fig.add_subplot(gs[0, 0])
    m = [null.Q_real.mean(), null.Q_nullA_mean.mean(), null.Q_nullB_mean.mean()]
    e = [null.Q_real.sem(), null.Q_nullA_mean.sem(), null.Q_nullB_mean.sem()]
    lbl = ["Observed", "Null A\n(rewired)", "Null B\n(permuted)"]
    cls = [JS.C_OURS, "#8491B4", "#B0BEC5"]
    ax.bar(range(3), m, yerr=e, color=cls, width=0.58, edgecolor="white",
           linewidth=0.5, capsize=2.4, error_kw=dict(elinewidth=0.6), zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(lbl, fontsize=5.2)
    ax.set_ylabel("Modularity $Q$", fontsize=6)
    ax.set_title("Observed vs null modularity", fontsize=6.5, pad=3)
    ax.text(0.5, 0.93, f"$z$ = {null.z_nullA.mean():.2f}",
            transform=ax.transAxes, fontsize=5.2, ha="center", va="top",
            color=JS.C_OURS)
    JS.tidy_axis(ax, grid_axis="y"); JS.panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(null.z_nullA.dropna(), bins=12, color="#8491B4",
            edgecolor="white", linewidth=0.5, alpha=0.9, label="Null A", zorder=3)
    ax.hist(null.z_nullB.dropna(), bins=12, color="#B0BEC5",
            edgecolor="white", linewidth=0.5, alpha=0.75, label="Null B", zorder=2)
    ax.axvline(0, color="#333333", lw=0.8, zorder=4)
    ax.set_xlabel("$z$-score vs null", fontsize=6)
    ax.set_ylabel("Count", fontsize=6)
    ax.set_title("Null-model $z$-scores", fontsize=6.5, pad=3)
    ax.legend(fontsize=5.0, loc="upper left")
    JS.tidy_axis(ax, grid_axis="y"); JS.panel_label(ax, "b", x=-0.26)

    o = FIG / "Fig6_null.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")


def fig7(dw):
    fig = plt.figure(figsize=(JS.ONE_HALF, 2.45))
    gs = fig.add_gridspec(1, 2, wspace=0.50, left=0.185, right=0.985,
                          top=0.855, bottom=0.175)

    agg = dw.groupby("algo")["AUC"].mean().sort_values(ascending=False)
    top = agg.head(10).iloc[::-1]
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(top))
    for yi, (nm, v) in zip(y, top.items()):
        c = JS.METHOD_COLORS.get(str(nm), None)
        if c is None:
            c = JS.C_OURS if str(nm).startswith("Leiden") else "#B0BEC5"
        ax.plot([0.5, v], [yi, yi], color=c, lw=1.4, solid_capstyle="round")
        ax.scatter([v], [yi], color=c, s=16, edgecolors="white", linewidths=0.4)
    ax.axvline(0.5, color="#BBBBBB", lw=0.6, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([JS.short_algo(n) for n in top.index], fontsize=4.7)
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("Cross-validated AUC", fontsize=6)
    ax.set_title("Downstream prediction", fontsize=6.5, pad=3)
    JS.tidy_axis(ax, grid_axis="x"); JS.panel_label(ax, "a", x=-0.68)

    ax = fig.add_subplot(gs[0, 1])
    cnt = dw.groupby("task").size()
    ax.barh(range(len(cnt)), cnt.values, color=JS.C_LEIDEN, height=0.6,
            edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yticks(range(len(cnt)))
    ax.set_yticklabels([str(t)[:20] for t in cnt.index], fontsize=4.7)
    ax.set_xlabel("Evaluation records", fontsize=6)
    ax.set_title("Task coverage", fontsize=6.5, pad=3)
    JS.tidy_axis(ax, grid_axis="x"); JS.panel_label(ax, "b", x=-0.52)

    o = FIG / "Fig7_downstream.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")


def main():
    fig5(pd.read_csv(OUT / "synthetic_results.csv"))
    fig6(pd.read_csv(OUT / "null_models.csv"))
    fig7(pd.read_csv(OUT / "downstream_v3.csv"))


if __name__ == "__main__":
    main()
