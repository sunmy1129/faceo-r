#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 12 (期刊级): 真实单细胞验证"""
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
from scipy.stats import wilcoxon

OUT = Path("/Users/sunmy/Documents/QF/GWAS/output/core_results")
FIG = Path("/Users/sunmy/Documents/QF/GWAS/output/figures_v3")
JS.apply_style()


def main():
    d = pd.read_csv(OUT / "scrna_validation.csv")
    piv = d.pivot_table(index="pathway", columns="algo", values="coh").dropna()
    drop = d.groupby("pathway")["dropout"].first()

    fig = plt.figure(figsize=(JS.DOUBLE, 2.5))
    gs = fig.add_gridspec(1, 4, wspace=0.52, left=0.055, right=0.99,
                          top=0.86, bottom=0.175, width_ratios=[1, 1, 1.15, 0.85])

    # (a) 真实 dropout 分布
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(drop.values * 100, bins=10, color=JS.NPG[5], edgecolor="white",
            linewidth=0.5, zorder=3)
    ax.axvline(50, color=JS.C_OURS, ls="--", lw=0.8, zorder=4)
    ax.text(51, ax.get_ylim()[1] * 0.82, "simulated\n(50%)", fontsize=4.4,
            color=JS.C_OURS, va="top")
    ax.set_xlabel("Zero fraction (%)", fontsize=6)
    ax.set_ylabel("Pathways", fontsize=6)
    ax.set_title("True scRNA dropout", fontsize=6.4, pad=3)
    JS.tidy_axis(ax, grid_axis="y"); JS.panel_label(ax, "a")

    # (b) 三方法配对斜率
    ax = fig.add_subplot(gs[0, 1])
    order = ["WGCNA", "Leiden", "FACEO-R"]
    cols = [JS.C_WGCNA, JS.C_LEIDEN, JS.C_OURS]
    for _, r in piv.iterrows():
        ax.plot(range(3), [r[o] for o in order], color="#B8B8B8", lw=0.4,
                alpha=0.45, zorder=1)
    means = [piv[o].mean() for o in order]
    sems = [piv[o].sem() for o in order]
    ax.errorbar(range(3), means, yerr=sems, color="black", lw=1.0, marker="o",
                ms=3.5, capsize=2, zorder=5, markeredgecolor="white",
                markeredgewidth=0.4)
    for x, m, c in zip(range(3), means, cols):
        ax.scatter([x], [m], color=c, s=22, zorder=6, edgecolors="white",
                   linewidths=0.4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["WGCNA", "Leiden", "FACEO-R"], fontsize=4.9)
    ax.set_ylabel("Coherence", fontsize=6)
    ax.set_xlim(-0.35, 2.35)
    ax.set_title(f"scRNA (n={len(piv)})", fontsize=6.4, pad=3)
    JS.tidy_axis(ax, grid_axis="y"); JS.panel_label(ax, "b", x=-0.30)

    # (c) 优势 vs dropout（关键：dropout 越高优势越大）
    ax = fig.add_subplot(gs[0, 2])
    adv_l = piv["FACEO-R"] - piv["Leiden"]
    adv_w = piv["FACEO-R"] - piv["WGCNA"]
    x = drop.reindex(piv.index).values * 100
    ax.axhline(0, color="#888888", lw=0.6, ls=":", zorder=1)
    ax.scatter(x, adv_l.values, s=14, color=JS.C_LEIDEN, alpha=0.9,
               edgecolors="white", linewidths=0.35, zorder=3, label="vs Leiden")
    ax.scatter(x, adv_w.values, s=14, color=JS.C_WGCNA, alpha=0.9,
               edgecolors="white", linewidths=0.35, zorder=3, label="vs WGCNA")
    for yv, c in [(adv_l.values, JS.C_LEIDEN), (adv_w.values, JS.C_WGCNA)]:
        if np.std(x) > 0:
            b = np.polyfit(x, yv, 1)
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(xs, np.polyval(b, xs), color=c, lw=0.75, ls="--", zorder=4)
    from scipy.stats import spearmanr
    rho, prho = spearmanr(x, adv_l.values)
    ax.set_xlabel("Zero fraction (%)", fontsize=6)
    ax.set_ylabel("Coherence gain", fontsize=6)
    ax.set_title(f"Gain across dropout levels\n($\\rho$={rho:+.2f}, n.s.)", fontsize=6.0, pad=3)
    ax.legend(fontsize=4.4, loc="upper left")
    JS.tidy_axis(ax, grid_axis="both"); JS.panel_label(ax, "c", x=-0.26)

    # (d) 胜率 + p
    ax = fig.add_subplot(gs[0, 3])
    res = []
    for ref, c in [("Leiden", JS.C_LEIDEN), ("WGCNA", JS.C_WGCNA)]:
        diff = piv["FACEO-R"] - piv[ref]
        res.append(((diff > 0).mean() * 100, wilcoxon(diff).pvalue if diff.std() > 0 else 1, c))
    y = [1, 0]
    for yi, (wr, pv, c) in zip(y, res):
        ax.barh(yi, wr, color=c, height=0.5, edgecolor="white", linewidth=0.4, zorder=3)
        ax.text(wr + 2, yi, f"{wr:.0f}%", fontsize=5.4, va="center", color=c)
        ax.text(2, yi - 0.36, f"$p$={pv:.1e}", fontsize=4.4, va="center",
                color="#666666")
    ax.set_yticks(y); ax.set_yticklabels(["vs\nLeiden", "vs\nWGCNA"], fontsize=4.9)
    ax.set_xlim(0, 108); ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Win rate (%)", fontsize=6)
    ax.set_title("Real scRNA\nadvantage", fontsize=6.4, pad=3)
    JS.tidy_axis(ax, grid_axis="x"); JS.panel_label(ax, "d", x=-0.42)

    o = FIG / "Fig12_scrna_validation.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")
    print(f"   FACEO-R {means[2]:.4f} vs Leiden {means[1]:.4f} vs WGCNA {means[0]:.4f}")
    print(f"   dropout 中位 {np.median(x):.1f}%, 与增益相关 ρ={rho:+.2f}")


if __name__ == "__main__":
    main()
