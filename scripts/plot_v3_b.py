#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig1-2：过分割现象 + 多指标对比"""
import sys
from pathlib import Path
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from plot_v3_a import load, agg, C, pl, W1, W2, OUT, HL


def fig1(df):
    g = df.groupby("algo").agg(
        n=("n_modules", "mean"), n_sd=("n_modules", "std"),
        Q=("modularity_Q", "mean"), Q_sd=("modularity_Q", "std"),
        z=("coh_z_mean", "mean"), z_sd=("coh_z_mean", "std")).reset_index()
    fig = plt.figure(figsize=(W2, 2.4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.45)

    # (a) 模块数 vs 模块度
    ax = fig.add_subplot(gs[0])
    for _, r in g.iterrows():
        c = C(r["algo"])
        ax.errorbar(r["n"], r["Q"], xerr=r["n_sd"], yerr=r["Q_sd"], fmt="o",
                    ms=3.5, color=c, ecolor=c, elinewidth=0.6, capsize=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Number of modules (log)")
    ax.set_ylabel("Modularity $Q$")
    ax.set_title("Fragmentation cost")
    for nm, lab in [("Leiden γ=0.5", "γ=0.5"), ("Leiden γ=1.0", "γ=1.0"),
                    ("Leiden γ=2.0", "γ=2.0"), ("Leiden γ=5.0", "γ=5.0"),
                    ("WGCNA-style Hierarchical", "WGCNA")]:
        r = g[g.algo == nm]
        if len(r):
            ax.annotate(lab, (r.iloc[0]["n"], r.iloc[0]["Q"]), fontsize=5.4,
                        xytext=(3, 3), textcoords="offset points", color=C(nm))
    pl(ax, "a")

    # (b) 模块数 vs 功能内聚
    ax2 = fig.add_subplot(gs[1])
    for _, r in g.iterrows():
        ax2.errorbar(r["n"], r["z"], xerr=r["n_sd"], yerr=r["z_sd"], fmt="o",
                     ms=3.5, color=C(r["algo"]), ecolor=C(r["algo"]),
                     elinewidth=0.6, capsize=1.2, zorder=3)
    ax2.axhline(0, ls=":", color="#999999", lw=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of modules (log)")
    ax2.set_ylabel("Functional coherence $z$")
    ax2.set_title("More modules ≠ more coherent")
    pl(ax2, "b")

    # (c) 分辨率扫描
    ax3 = fig.add_subplot(gs[2])
    lg = df[df.algo.str.startswith("Leiden γ=")]
    piv = lg.groupby("algo").agg(q=("modularity_Q", "mean"),
                                 z=("coh_z_mean", "mean"),
                                 n=("n_modules", "mean")).reset_index()
    piv["gamma"] = piv.algo.str.extract(r"γ=([0-9.]+)").astype(float)
    piv = piv.sort_values("gamma")
    ax3.plot(piv["gamma"], piv["q"], "o-", color="#D55E00", ms=3.5, lw=1.0)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Resolution $\\gamma$")
    ax3.set_ylabel("Modularity $Q$", color="#D55E00")
    ax3.tick_params(axis="y", labelcolor="#D55E00")
    ax3.set_title("Resolution collapse")
    ax3b = ax3.twinx()
    ax3b.plot(piv["gamma"], piv["z"], "s--", color="#009E73", ms=3.2, lw=1.0)
    ax3b.set_ylabel("Coherence $z$", color="#009E73")
    ax3b.tick_params(axis="y", labelcolor="#009E73")
    ax3b.spines["right"].set_visible(True)
    ax3b.spines["top"].set_visible(False)
    pl(ax3, "c")
    o = OUT / "Fig1_overpartition.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    return o


def fig2(df):
    a = df.groupby("algo").agg(
        z=("coh_z_mean", "mean"), zs=("coh_z_mean", "std"),
        stab=("stability_ARI", "mean"), rob=("ARI_noise0.1", "mean"),
        ant=("antag_split_rate", "mean"), n=("n_modules", "count")).dropna(subset=["z"])
    a = a[a["n"] >= a["n"].max() * 0.85].sort_values("z", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.5))
    fig.subplots_adjust(wspace=0.55)

    ax = axes[0]
    y = np.arange(len(a))
    ax.barh(y, a["z"], xerr=a["zs"].fillna(0), height=0.7,
            color=[C(i) for i in a.index], error_kw=dict(lw=0.5, capsize=1))
    ax.set_yticks(y); ax.set_yticklabels(a.index, fontsize=5.2)
    ax.invert_yaxis()
    ax.axvline(0, ls=":", color="#999999", lw=0.6)
    ax.set_xlabel("Functional coherence $z$")
    pl(ax, "a", x=-0.55)

    ax2 = axes[1]
    for i, r in a.iterrows():
        ax2.scatter(r["stab"], r["rob"], s=16, color=C(i), zorder=3,
                    edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Stability (bootstrap ARI)")
    ax2.set_ylabel("Robustness (ARI, 10% noise)")
    pl(ax2, "b")

    ax3 = axes[2]
    b = a.dropna(subset=["ant"]).sort_values("ant", ascending=False)
    x = np.arange(len(b))
    ax3.bar(x, b["ant"], color=[C(i) for i in b.index], width=0.7)
    ax3.set_xticks(x); ax3.set_xticklabels(b.index, rotation=90, fontsize=4.8)
    ax3.set_ylabel("Antagonistic pair\nseparation rate")
    ax3.set_ylim(0, 1.08)
    pl(ax3, "c", x=-0.25)
    o = OUT / "Fig2_metrics.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    return o


if __name__ == "__main__":
    df = load()
    print(f"载入 {len(df)} 条 | 数据集 {df.gse.nunique()} | 算法 {df.algo.nunique()}")
    print("✅", fig1(df).name)
    print("✅", fig2(df).name)
