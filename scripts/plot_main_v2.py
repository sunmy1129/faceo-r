#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 1 & 2 (期刊级重绘): 过分割现象 + 多指标对比"""
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
from scipy.stats import spearmanr, wilcoxon

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
OUT = BASE / "output/core_results"
FIG = BASE / "output/figures_v3"
JS.apply_style()


def fig1(df):
    """过分割：模块数 vs 模块度 / 功能一致性 + 分辨率扫描"""
    lg = df[df.algo.astype(str).str.startswith("Leiden γ=")].copy()
    lg["gamma"] = lg.algo.astype(str).str.extract(r"γ=([0-9.]+)").astype(float)

    fig = plt.figure(figsize=(JS.DOUBLE, 2.5))
    gs = fig.add_gridspec(1, 3, wspace=0.40, left=0.062, right=0.99,
                          top=0.87, bottom=0.165)

    # (a) 模块数 vs 模块度
    ax = fig.add_subplot(gs[0, 0])
    x = df["n_modules"].values; y = df["modularity_Q"].values
    rho, p = spearmanr(x, y)
    ax.scatter(x, y, s=3.2, alpha=0.42, color=JS.NPG[0], edgecolors="none", zorder=2)
    m = (x > 0) & (y > -1)
    if m.sum() > 10:
        b = np.polyfit(x[m], y[m], 1)
        xs = np.linspace(x[m].min(), np.percentile(x[m], 99), 40)
        ax.plot(xs, np.polyval(b, xs), color="#333333", lw=0.9, ls="--", zorder=4)
    ax.set_xlabel("Number of modules", fontsize=6)
    ax.set_ylabel("Modularity $Q$", fontsize=6)
    ax.text(0.96, 0.93, f"$\\rho$ = {rho:+.2f}\n$p$ < 0.001" if p < 0.001
            else f"$\\rho$ = {rho:+.2f}", transform=ax.transAxes, fontsize=5.0,
            ha="right", va="top", color="#333333")
    JS.tidy_axis(ax, grid_axis="both"); JS.panel_label(ax, "a")

    # (b) 模块数 vs 功能一致性
    ax = fig.add_subplot(gs[0, 1])
    x = df["n_modules"].values; y = df["coh_z_mean"].values
    ok = np.isfinite(y)
    ax.scatter(x[ok], y[ok], s=3.2, alpha=0.42, color=JS.NPG[2],
               edgecolors="none", zorder=2)
    rho2, p2 = spearmanr(x[ok], y[ok])
    ax.axhline(0, color="#BBBBBB", lw=0.6, ls=":", zorder=1)
    ax.set_xlabel("Number of modules", fontsize=6)
    ax.set_ylabel("Functional coherence $z$", fontsize=6)
    ax.text(0.96, 0.93, f"$\\rho$ = {rho2:+.2f}\n$p$ = {p2:.1e}", transform=ax.transAxes,
            fontsize=5.0, ha="right", va="top", color="#333333")
    JS.tidy_axis(ax, grid_axis="both"); JS.panel_label(ax, "b")

    # (c) 分辨率扫描
    ax = fig.add_subplot(gs[0, 2])
    g = lg.groupby("gamma").agg(nmod=("n_modules", "median"),
                                Q=("modularity_Q", "median"),
                                coh=("coh_z_mean", "median")).reset_index()
    ax.plot(g.gamma, g.Q, color=JS.C_LEIDEN, marker="o", ms=3, lw=0.9,
            label="Modularity $Q$", markeredgecolor="white", markeredgewidth=0.35)
    ax.set_xlabel("Resolution $\\gamma$", fontsize=6)
    ax.set_ylabel("Modularity $Q$", fontsize=6, color=JS.C_LEIDEN)
    ax.set_xscale("log")
    ax.tick_params(axis="y", colors=JS.C_LEIDEN)
    ax2 = ax.twinx()
    ax2.plot(g.gamma, g.coh, color=JS.C_OURS, marker="s", ms=3, lw=0.9,
             ls="--", label="Coherence $z$", markeredgecolor="white",
             markeredgewidth=0.35)
    ax2.set_ylabel("Coherence $z$", fontsize=6, color=JS.C_OURS)
    ax2.tick_params(axis="y", colors=JS.C_OURS)
    for s in ["top"]:
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_linewidth(0.5)
    ax2.spines["right"].set_color(JS.C_OURS)
    ax.axvspan(1.0, 2.0, color="#F0F0F0", zorder=0)
    ax.text(1.41, ax.get_ylim()[1] * 0.96, "optimal\nrange", fontsize=4.6,
            ha="center", va="top", color="#888888")
    JS.tidy_axis(ax, grid_axis="y"); JS.panel_label(ax, "c", x=-0.20)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)

    o = FIG / "Fig1_overpartition.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")


def fig2(df):
    """多指标：功能一致性排名 + 稳定性-鲁棒性空间 + 拮抗拆分"""
    agg = df.groupby("algo").agg(
        coh=("coh_z_mean", "mean"), stab=("stability_ARI", "mean"),
        rob=("ARI_noise0.1", "mean"), ant=("antag_split_rate", "mean"),
        nmod=("n_modules", "mean")).dropna(subset=["coh"])
    agg = agg.sort_values("coh", ascending=False)

    fig = plt.figure(figsize=(JS.DOUBLE, 2.75))
    gs = fig.add_gridspec(1, 3, wspace=0.50, left=0.155, right=0.985,
                          top=0.87, bottom=0.155)

    # (a) 排名 lollipop（竖排）
    ax = fig.add_subplot(gs[0, 0])
    top = agg.head(14).iloc[::-1]
    y = np.arange(len(top))
    for yi, (nm, r) in zip(y, top.iterrows()):
        col = JS.METHOD_COLORS.get(nm, None)
        if col is None:
            col = JS.C_OURS if nm.startswith("FACEO") else "#B0BEC5"
        ax.plot([0, r["coh"]], [yi, yi], color=col, lw=1.5,
                solid_capstyle="round", zorder=2)
        ax.scatter([r["coh"]], [yi], color=col, s=18, zorder=3,
                   edgecolors="white", linewidths=0.4)
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([JS.short_algo(n) for n in top.index], fontsize=4.7)
    ax.set_xlabel("Functional coherence $z$", fontsize=6)
    ax.set_title("Top algorithms by coherence", fontsize=6.5, pad=3)
    JS.tidy_axis(ax, grid_axis="x"); JS.panel_label(ax, "a", x=-0.62)

    # (b) 稳定性 vs 鲁棒性
    ax = fig.add_subplot(gs[0, 1])
    s = agg.dropna(subset=["stab", "rob"])
    for nm, r in s.iterrows():
        col = JS.METHOD_COLORS.get(nm, "#B0BEC5")
        ax.scatter(r["stab"], r["rob"], s=13, color=col, alpha=0.85,
                   edgecolors="white", linewidths=0.3, zorder=3)
    ax.set_xlabel("Stability (bootstrap ARI)", fontsize=6)
    ax.set_ylabel("Robustness (ARI, 10% noise)", fontsize=6)
    ax.set_title("Stability–robustness space", fontsize=6.5, pad=3)
    JS.tidy_axis(ax, grid_axis="both"); JS.panel_label(ax, "b", x=-0.28)

    # (c) 拮抗基因拆分率
    ax = fig.add_subplot(gs[0, 2])
    a = agg.dropna(subset=["ant"]).sort_values("ant", ascending=False).head(12)
    y = np.arange(len(a))[::-1]
    cols = [JS.METHOD_COLORS.get(n, "#B0BEC5") for n in a.index]
    ax.barh(y, a["ant"].values, color=cols, height=0.62,
            edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([JS.short_algo(n) for n in a.index], fontsize=4.7)
    ax.set_xlabel("Antagonistic pair separation", fontsize=6)
    ax.set_xlim(0, 1.02)
    ax.set_title("Functionally opposed genes", fontsize=6.5, pad=3)
    JS.tidy_axis(ax, grid_axis="x"); JS.panel_label(ax, "c", x=-0.56)

    o = FIG / "Fig2_metrics.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")


def main():
    df = pd.read_csv(OUT / "full_eval_v3.csv")
    fig1(df); fig2(df)


if __name__ == "__main__":
    main()
