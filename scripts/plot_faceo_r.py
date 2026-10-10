#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FACEO-R 结果图：降质鲁棒性曲线 + 胜率 + 自适应权重机制"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
BASE = Path("/Users/sunmy/Documents/QF/GWAS")
OUT = BASE / "output/core_results"
FIG = BASE / "output/figures_v3"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.6,
    "axes.linewidth": 0.6, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_R, C_L, C_W = "#D55E00", "#0072B2", "#666666"


def main():
    d = pd.read_csv(OUT / "faceo_r_results.csv")
    d["kind"] = d.condition.str.rsplit("_", n=1).str[0]
    d["lvl"] = d.condition.str.rsplit("_", n=1).str[1].astype(float)
    NICE = {"samples": "Sample subsampling", "noise": "Expression noise",
            "dropout": "Feature dropout"}

    fig = plt.figure(figsize=(183 * MM, 4.6))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.42)

    # (a-c) 每种降质条件的曲线
    for ai, kind in enumerate(["dropout", "noise", "samples"]):
        ax = fig.add_subplot(gs[0, ai])
        sub = d[d.kind == kind]
        for algo, col, mk in [("FACEO-R (ours)", C_R, "o"),
                              ("Leiden γ=1.0", C_L, "s"),
                              ("WGCNA-style", C_W, "^")]:
            g = sub[sub.algo == algo].groupby("lvl")["coh"].agg(["mean", "sem"])
            if len(g):
                ax.errorbar(g.index, g["mean"], yerr=g["sem"].fillna(0), label=algo,
                            color=col, marker=mk, ms=3, lw=1.1, capsize=1.8,
                            elinewidth=0.6)
        ax.set_xlabel({"dropout": "Dropout fraction", "noise": "Noise (× SD)",
                       "samples": "Sample fraction retained"}[kind], fontsize=6.8)
        if ai == 0:
            ax.set_ylabel("Functional coherence\n(held-out Reactome)", fontsize=6.8)
        ax.set_title(NICE[kind], fontsize=7.0)
        ax.tick_params(labelsize=6)
        if kind == "samples":
            ax.invert_xaxis()
        ax.text(-0.30, 1.06, "abc"[ai], transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")
        if ai == 0:
            ax.legend(fontsize=5.2, frameon=False, loc="best")

    # (d) 胜率
    ax = fig.add_subplot(gs[1, 0])
    conds = ["dropout_0.5", "noise_1.0", "noise_0.5", "samples_0.3", "samples_0.15"]
    wl, ww = [], []
    for c in conds:
        p = d[d.condition == c].pivot_table(index=["gse", "pathway"],
                                            columns="algo", values="coh").dropna()
        wl.append((p["FACEO-R (ours)"] > p["Leiden γ=1.0"]).mean() * 100)
        ww.append((p["FACEO-R (ours)"] > p["WGCNA-style"]).mean() * 100)
    x = np.arange(len(conds))
    ax.bar(x - 0.19, wl, 0.36, color=C_L, label="vs Leiden")
    ax.bar(x + 0.19, ww, 0.36, color=C_W, label="vs WGCNA")
    ax.axhline(50, ls=":", color="#999999", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=5.4)
    ax.set_ylabel("Win rate (%)", fontsize=6.8)
    ax.set_ylim(0, 105)
    ax.set_title("Per-condition win rate", fontsize=7.0)
    ax.legend(fontsize=5.2, frameon=False)
    ax.tick_params(labelsize=6)
    ax.text(-0.30, 1.06, "d", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")

    # (e) 自适应权重机制
    ax = fig.add_subplot(gs[1, 1])
    q = d.groupby("condition")["reliability"].first()
    a = d.groupby("condition")["alpha"].first()
    ax.scatter(q, a, c=C_R, s=26, zorder=3, edgecolors="white", linewidths=0.4)
    z = np.polyfit(q, a, 1)
    xs = np.linspace(q.min(), q.max(), 20)
    ax.plot(xs, np.polyval(z, xs), ls="--", color="#999999", lw=0.8)
    ax.set_xlabel("Graph reliability $q$", fontsize=6.8)
    ax.set_ylabel("Adaptive weight $\\alpha$", fontsize=6.8)
    ax.set_title("Anchoring mechanism", fontsize=7.0)
    ax.tick_params(labelsize=6)
    ax.text(0.03, 0.92, "$\\alpha(q)=\\alpha_{min}+(\\alpha_{max}-\\alpha_{min})q$",
            transform=ax.transAxes, fontsize=5.4, va="top")
    ax.text(-0.30, 1.06, "e", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")

    # (f) 全局汇总
    ax = fig.add_subplot(gs[1, 2])
    p = d.pivot_table(index=["gse", "pathway", "condition"], columns="algo",
                      values="coh").dropna()
    means = [p["FACEO-R (ours)"].mean(), p["Leiden γ=1.0"].mean(),
             p["WGCNA-style"].mean()]
    errs = [p["FACEO-R (ours)"].sem(), p["Leiden γ=1.0"].sem(),
            p["WGCNA-style"].sem()]
    cols = [C_R, C_L, C_W]
    lbl = ["FACEO-R\n(ours)", "Leiden\nγ=1.0", "WGCNA"]
    ax.bar(range(3), means, yerr=errs, color=cols, width=0.6,
           capsize=2.5, error_kw=dict(elinewidth=0.7))
    ax.set_xticks(range(3)); ax.set_xticklabels(lbl, fontsize=5.6)
    ax.set_ylabel("Mean coherence", fontsize=6.8)
    ax.set_title(f"Overall (n={len(p)} pairs)", fontsize=7.0)
    ax.tick_params(labelsize=6)
    from scipy.stats import wilcoxon
    pv = wilcoxon(p["FACEO-R (ours)"] - p["Leiden γ=1.0"]).pvalue
    ax.text(0.5, 0.94, f"$p$ = {pv:.1e}", transform=ax.transAxes,
            fontsize=5.8, ha="center", va="top")
    ax.text(-0.30, 1.06, "f", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")

    o = FIG / "Fig11_faceo_r_robustness.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")
    print(f"全局: FACEO-R {means[0]:.4f} vs Leiden {means[1]:.4f} vs WGCNA {means[2]:.4f}")
    print(f"p = {pv:.2e}")


if __name__ == "__main__":
    main()
