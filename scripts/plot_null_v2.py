#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 6 v2 (期刊级): 模块度对图构建的敏感性（替代原零模型图）"""
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

OUT = Path("/Users/sunmy/Documents/QF/GWAS/output/core_results")
FIG = Path("/Users/sunmy/Documents/QF/GWAS/output/figures_v3")
JS.apply_style()


def main():
    d = pd.read_csv(OUT / "weight_dependence.csv")

    fig = plt.figure(figsize=(JS.ONE_HALF, 2.4))
    gs = fig.add_gridspec(1, 2, wspace=0.46, left=0.125, right=0.985,
                          top=0.855, bottom=0.175)

    # (a) 阈值敏感性：Q 随阈值单调上升
    ax = fig.add_subplot(gs[0, 0])
    agg = d.groupby("threshold").agg(
        qw=("Q_weighted", "mean"), qwsem=("Q_weighted", "sem")).reset_index()
    ax.errorbar(agg.threshold, agg.qw, yerr=agg.qwsem, color=JS.C_OURS,
                marker="o", ms=3.2, lw=1.0, capsize=2.2,
                markeredgecolor="white", markeredgewidth=0.4, zorder=4)
    ax.fill_between(agg.threshold, agg.qw - agg.qwsem, agg.qw + agg.qwsem,
                    color=JS.C_OURS, alpha=0.13, lw=0)
    ax.axvline(75, color=JS.C_LEIDEN, ls="--", lw=0.75, zorder=2)
    ax.text(76, agg.qw.min(), "common\nchoice", fontsize=4.5, color=JS.C_LEIDEN,
            va="bottom")
    ax.annotate("", xy=(95, agg.qw.iloc[-1]), xytext=(50, agg.qw.iloc[0]),
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#999999",
                                connectionstyle="arc3,rad=-0.25"), zorder=1)
    ax.set_xlabel("Edge-weight threshold (percentile)", fontsize=6)
    ax.set_ylabel("Modularity $Q$", fontsize=6)
    ax.set_title("Modularity depends on construction", fontsize=6.4, pad=3)
    ax.text(0.03, 0.94, "$\\Delta Q$ = +56%", transform=ax.transAxes,
            fontsize=5.2, va="top", color=JS.C_OURS)
    JS.tidy_axis(ax, grid_axis="y"); JS.panel_label(ax, "a")

    # (b) 权重贡献
    ax = fig.add_subplot(gs[0, 1])
    thresholds = sorted(d.threshold.unique())
    xs = np.arange(len(thresholds))
    qw = [d[d.threshold == t].Q_weighted.mean() for t in thresholds]
    qu = [d[d.threshold == t].Q_uniform.mean() for t in thresholds]
    ax.bar(xs - 0.19, qw, 0.36, color=JS.C_OURS, label="weighted",
           edgecolor="white", linewidth=0.4, zorder=3)
    ax.bar(xs + 0.19, qu, 0.36, color="#B8C4D9", label="uniform weights",
           edgecolor="white", linewidth=0.4, zorder=3)
    for x, a, b in zip(xs, qw, qu):
        ax.text(x, max(a, b) + 0.012, f"{(a-b)/a*100:.0f}%", fontsize=4.4,
                ha="center", color="#666666")
    ax.set_xticks(xs); ax.set_xticklabels([f"{t}" for t in thresholds], fontsize=5.2)
    ax.set_xlabel("Threshold (percentile)", fontsize=6)
    ax.set_ylabel("Modularity $Q$", fontsize=6)
    ax.set_title("Weight contribution (29% overall)", fontsize=6.4, pad=3)
    ax.set_ylim(0, max(qw) * 1.22)
    ax.legend(fontsize=4.8, loc="upper left")
    JS.tidy_axis(ax, grid_axis="y"); JS.panel_label(ax, "b", x=-0.24)

    o = FIG / "Fig6_null.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name} (阈值敏感性 + 权重贡献)")
    print(f"   Q: 阈值50→95 从 {qw[0]:.3f} 升到 {qw[-1]:.3f} (+{(qw[-1]-qw[0])/qw[0]*100:.0f}%)")
    print(f"   权重贡献: 加权 {np.mean(qw):.3f} vs 均匀 {np.mean(qu):.3f} "
          f"({(np.mean(qw)-np.mean(qu))/np.mean(qw)*100:.0f}%)")


if __name__ == "__main__":
    main()
