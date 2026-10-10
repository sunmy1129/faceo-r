#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 11 (期刊级重绘): FACEO-R 降质鲁棒性"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

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

METH = [("FACEO-R (ours)", JS.C_OURS, "o", "-"),
        ("Leiden γ=1.0", JS.C_LEIDEN, "s", "-"),
        ("WGCNA-style", JS.C_WGCNA, "^", "--")]
KINDS = [("dropout", "Feature dropout", "Dropout fraction", False),
         ("noise", "Expression noise", "Noise ($\\times\\sigma$)", False),
         ("samples", "Sample subsampling", "Fraction retained", True)]


def main():
    d = pd.read_csv(OUT / "faceo_r_results.csv")
    d["kind"] = d.condition.str.rsplit("_", n=1).str[0]
    d["lvl"] = d.condition.str.rsplit("_", n=1).str[1].astype(float)
    piv_all = d.pivot_table(index=["gse", "pathway", "condition"],
                            columns="algo", values="coh").dropna()

    fig = plt.figure(figsize=(JS.DOUBLE, 4.35))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.46,
                          left=0.055, right=0.99, top=0.93, bottom=0.075)

    # ---- (a–c) 降质曲线：误差带 + 数据点 ----
    for ai, (kind, title, xlab, inv) in enumerate(KINDS):
        ax = fig.add_subplot(gs[0, ai])
        sub = d[d.kind == kind]
        for algo, col, mk, ls in METH:
            g = sub[sub.algo == algo].groupby("lvl")["coh"].agg(["mean", "sem", "count"])
            if not len(g):
                continue
            x, y = g.index.values, g["mean"].values
            se = g["sem"].fillna(0).values
            ax.fill_between(x, y - se, y + se, color=col, alpha=0.16, lw=0)
            ax.plot(x, y, color=col, marker=mk, ms=2.6, lw=0.85, ls=ls,
                    markeredgecolor="white", markeredgewidth=0.3, zorder=3,
                    label=algo)
        ax.set_xlabel(xlab, fontsize=6)
        if ai == 0:
            ax.set_ylabel("Functional coherence\n(held-out Reactome)", fontsize=6)
        ax.set_title(title, fontsize=6.6, pad=3.5)
        if inv:
            ax.invert_xaxis()
        JS.tidy_axis(ax, grid_axis="y")
        JS.panel_label(ax, "abc"[ai])
        if ai == 0:
            ax.legend(loc="lower left", fontsize=4.9, borderpad=0.15,
                      handletextpad=0.5, labelspacing=0.22)

    # ---- (d) 胜率 lollipop ----
    ax = fig.add_subplot(gs[1, 0])
    conds = ["dropout_0.5", "noise_0.5", "noise_1.0", "samples_0.3", "samples_0.15"]
    nice = ["Dropout\n0.5", "Noise\n0.5", "Noise\n1.0", "Subsamp.\n0.3", "Subsamp.\n0.15"]
    wl, ww = [], []
    for c in conds:
        p = d[d.condition == c].pivot_table(index=["gse", "pathway"],
                                            columns="algo", values="coh").dropna()
        wl.append((p["FACEO-R (ours)"] > p["Leiden γ=1.0"]).mean() * 100)
        ww.append((p["FACEO-R (ours)"] > p["WGCNA-style"]).mean() * 100)
    y = np.arange(len(conds))[::-1]
    for yi, a, b in zip(y, wl, ww):
        ax.plot([0, a], [yi + 0.16, yi + 0.16], color=JS.C_LEIDEN, lw=1.5,
                solid_capstyle="round", zorder=2)
        ax.plot([0, b], [yi - 0.16, yi - 0.16], color=JS.C_WGCNA, lw=1.5,
                solid_capstyle="round", zorder=2, ls=(0, (2, 1)))
        ax.scatter([a], [yi + 0.16], color=JS.C_LEIDEN, s=17, zorder=3,
                   edgecolors="white", linewidths=0.4)
        ax.scatter([b], [yi - 0.16], color=JS.C_WGCNA, s=17, zorder=3,
                   edgecolors="white", linewidths=0.4)
    ax.axvline(50, color="#BBBBBB", ls=":", lw=0.6, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(nice, fontsize=5.2)
    ax.set_xlim(0, 105); ax.set_ylim(-0.7, len(conds) - 0.3)
    ax.set_xlabel("Win rate (%)", fontsize=6)
    ax.set_title("Per-condition win rate", fontsize=6.6, pad=3.5)
    JS.tidy_axis(ax, grid_axis="x")
    JS.panel_label(ax, "d")
    ax.legend(handles=[Line2D([], [], color=JS.C_LEIDEN, lw=1.5, marker="o",
                              ms=3, label="vs Leiden"),
                       Line2D([], [], color=JS.C_WGCNA, lw=1.5, ls=(0, (2, 1)),
                              marker="o", ms=3, label="vs WGCNA")],
              loc="lower right", fontsize=4.9)

    # ---- (e) 自适应锚定机制 ----
    ax = fig.add_subplot(gs[1, 1])
    q = d.groupby("condition")["reliability"].first()
    a = d.groupby("condition")["alpha"].first()
    ax.scatter(q, a, s=24, color=JS.C_OURS, zorder=4, edgecolors="white",
               linewidths=0.5)
    if len(q) > 2:
        z = np.polyfit(q.values, a.values, 1)
        xs = np.linspace(q.min(), q.max(), 30)
        ax.plot(xs, np.polyval(z, xs), ls="--", color="#999999", lw=0.65, zorder=3)
    ax.set_xlabel("Graph reliability $q$", fontsize=6)
    ax.set_ylabel("Anchoring weight $\\alpha$", fontsize=6)
    ax.set_title("Adaptive mechanism", fontsize=6.6, pad=3.5)
    ax.text(0.04, 0.93,
            r"$\alpha(q)=\alpha_{\min}+(\alpha_{\max}-\alpha_{\min})q$",
            transform=ax.transAxes, fontsize=5.0, va="top", color="#444444")
    JS.tidy_axis(ax, grid_axis="both")
    JS.panel_label(ax, "e", x=-0.24)

    # ---- (f) 配对斜率图（比条形图信息量高）----
    ax = fig.add_subplot(gs[1, 2])
    order = ["WGCNA-style", "Leiden γ=1.0", "FACEO-R (ours)"]
    xs = [0, 1, 2]
    cols = [JS.C_WGCNA, JS.C_LEIDEN, JS.C_OURS]
    for idx, r in piv_all.iterrows():
        vals = [r.get(o, np.nan) for o in order]
        ax.plot(xs, vals, color="#BBBBBB", lw=0.35, alpha=0.35, zorder=1)
    means = [piv_all[o].mean() for o in order]
    sems = [piv_all[o].sem() for o in order]
    ax.errorbar(xs, means, yerr=sems, color="black", lw=1.1, marker="o",
                ms=4, capsize=2, zorder=5, markeredgecolor="white",
                markeredgewidth=0.5)
    for x, m, c in zip(xs, means, cols):
        ax.scatter([x], [m], color=c, s=26, zorder=6, edgecolors="white",
                   linewidths=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["WGCNA", "Leiden\n$\\gamma$=1.0", "FACEO-R\n(ours)"],
                       fontsize=5.2)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylabel("Coherence", fontsize=6)
    pv = wilcoxon(piv_all["FACEO-R (ours)"] - piv_all["Leiden γ=1.0"]).pvalue
    ax.set_title(f"Overall (n={len(piv_all)})", fontsize=6.6, pad=3.5)
    ax.text(0.97, 0.06, f"$p$ = {pv:.1e}", transform=ax.transAxes,
            fontsize=5.0, ha="right", va="bottom", color="#444444")
    JS.tidy_axis(ax, grid_axis="y")
    JS.panel_label(ax, "f", x=-0.24)

    o = FIG / "Fig11_faceo_r_robustness.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}  (期刊级重绘)")
    print(f"   全局均值: " + ", ".join(f"{o.split()[0]}={m:.4f}" for o, m in zip(order, means)))
    print(f"   p = {pv:.2e}")


if __name__ == "__main__":
    main()
