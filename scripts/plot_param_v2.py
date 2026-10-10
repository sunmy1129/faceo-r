#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 10 (期刊级重绘): 分辨率不可预测性"""
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
from scipy.stats import entropy

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
OUT = BASE / "output/core_results"
FIG = BASE / "output/figures_v3"
JS.apply_style()


def main():
    bt = pd.read_csv(OUT / "optimal_gamma_distribution.csv")
    at = pd.read_csv(OUT / "parameter_selection_attempts.csv")
    ent = entropy(bt["fraction"].values, base=np.e)
    ent_max = np.log(len(bt))

    fig = plt.figure(figsize=(JS.ONE_HALF, 2.55))
    gs = fig.add_gridspec(1, 2, wspace=0.42, left=0.085, right=0.985,
                          top=0.86, bottom=0.155)

    # ---- (a) 最优 γ 分布 ----
    ax = fig.add_subplot(gs[0, 0])
    g = bt["gamma"].values
    c = bt["count"].values
    cols = [JS.C_OURS if abs(x - 1.0) < 1e-6 else "#B8C4D9" for x in g]
    bars = ax.bar(range(len(g)), c, color=cols, width=0.66,
                  edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(c.mean(), ls=(0, (3, 2)), lw=0.6, color="#999999", zorder=1)
    ax.text(len(g) - 0.4, c.mean() + 0.5, "uniform\nexpectation",
            fontsize=4.6, ha="right", va="bottom", color="#777777")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{x:g}" for x in g], fontsize=5.5)
    ax.set_xlabel("Resolution $\\gamma$", fontsize=6)
    ax.set_ylabel("Pathways where optimal", fontsize=6)
    ax.set_ylim(0, c.max() * 1.30)
    for i, v in enumerate(c):
        ax.text(i, v + 0.35, str(v), ha="center", fontsize=5.2, color="#444444")
    ax.text(0.03, 0.95, f"$H$ = {ent:.2f} / {ent_max:.2f}  (near-uniform)",
            transform=ax.transAxes, fontsize=5.0, va="top", color="#333333")
    JS.tidy_axis(ax, grid_axis="y")
    JS.panel_label(ax, "a")

    # ---- (b) 四类策略全部失败 ----
    ax = fig.add_subplot(gs[0, 1])
    lbl = ["Structural\nfingerprint\n(RandomForest)", "Structural\nfingerprint\n(logistic)",
           "Pathway-size\nbinning", "Multi-resolution\nconsensus", "Cross-library\nheld-out"]
    gain = at.set_index("strategy")["gain_vs_fixed_gamma"].values
    verdict = at.set_index("strategy")["verdict"].values
    y = np.arange(len(lbl))[::-1]
    for yi, gv, vd in zip(y, gain, verdict):
        if gv != gv:  # NaN -> 无显著相关
            ax.scatter([0], [yi], facecolors="white", edgecolors="#999999",
                       s=26, zorder=4, linewidths=0.7)
            ax.text(0.06, yi, " n.s.  ($\\rho$=0.22, $p$=0.064)", fontsize=4.7,
                    va="center", color="#777777")
            continue
        col = JS.NPG[7] if gv < 0 else ("#4DBBD5" if abs(gv) < 0.4 else JS.NPG[2])
        ax.plot([0, gv], [yi, yi], color=col, lw=1.4, solid_capstyle="round",
                zorder=2)
        ax.scatter([gv], [yi], color=col, s=22, zorder=4,
                   edgecolors="white", linewidths=0.5)
        off = 0.06 if gv > 0 else -0.06
        ax.text(gv + off, yi, f"{gv:+.2f}", fontsize=4.9, va="center",
                ha="left" if gv > 0 else "right", color="#333333")
    ax.axvline(0, color="#555555", lw=0.6, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=4.8)
    ax.set_xlabel("Gain vs fixed $\\gamma$ = 1.0", fontsize=6)
    ax.set_xlim(-2.7, 1.0)
    ax.set_ylim(-0.7, len(lbl) - 0.3)
    JS.tidy_axis(ax, grid_axis="x")
    JS.panel_label(ax, "b", x=-0.52)
    ax.legend(handles=[
        Line2D([], [], color=JS.NPG[7], lw=1.4, marker="o", ms=3.2, label="worse than fixed"),
        Line2D([], [], color="#4DBBD5", lw=1.4, marker="o", ms=3.2, label="marginal"),
        Line2D([], [], color="#999999", lw=1.4, marker="o", ms=3.2, mfc="white",
               label="no association")],
        loc="lower left", fontsize=4.6, handlelength=1.4)

    o = FIG / "Fig10_parameter_selection.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}  (期刊级重绘)")


if __name__ == "__main__":
    main()
