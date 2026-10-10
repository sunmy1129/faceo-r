#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig3-4：性能画像热图 + 跨数据集一致性"""
import sys
from pathlib import Path
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_v3_a import load, C, pl, W1, W2, OUT, HL


def fig3(df):
    metrics = [c for c in ["n_modules", "modularity_Q", "coh_z_mean",
                           "stability_ARI", "ARI_noise0.1", "antag_split_rate"]
               if c in df.columns]
    piv = df.groupby("algo")[metrics].mean()
    top = df.groupby("algo")["coh_z_mean"].mean().sort_values(ascending=False)
    algos = list(top.index)
    arr = piv.loc[algos].values.astype(float)
    z = (arr - np.nanmean(arr, 0)) / (np.nanstd(arr, 0) + 1e-9)

    fig, ax = plt.subplots(figsize=(W1 * 1.5, 3.6))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=5.2)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=5.0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.2,
                        color="black" if abs(z[i, j]) < 1.5 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("$z$-score", fontsize=6)
    cb.ax.tick_params(labelsize=5.2)
    ax.set_title(f"Algorithm performance profile "
                 f"({df.gse.nunique()} cohorts, {df.groupby(['gse','pathway']).ngroups} pairs)",
                 fontsize=7)
    o = OUT / "Fig3_heatmap.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    return o


def fig4(df):
    """跨数据集一致性：每个数据集上实验组与对照的功能内聚差值"""
    d = df[df.algo.isin(["Leiden γ=1.0", "WGCNA-style Hierarchical"])]
    piv = d.pivot_table(index=["gse", "pathway"], columns="algo",
                        values="coh_z_mean").dropna()
    piv["diff"] = piv["Leiden γ=1.0"] - piv["WGCNA-style Hierarchical"]
    by_gse = piv.groupby(level=0)["diff"].agg(["mean", "sem", "count"]).sort_values("mean")
    wins = (by_gse["mean"] > 0).sum()

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4))
    fig.subplots_adjust(wspace=0.4)

    ax = axes[0]
    y = np.arange(len(by_gse))
    cols = ["#009E73" if v > 0 else "#D55E00" for v in by_gse["mean"]]
    ax.barh(y, by_gse["mean"], xerr=by_gse["sem"].fillna(0), height=0.7,
            color=cols, error_kw=dict(lw=0.5, capsize=1))
    ax.set_yticks(y); ax.set_yticklabels(by_gse.index, fontsize=5.6)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("$\\Delta$ functional coherence\n(Leiden γ=1.0 − WGCNA)")
    ax.set_title(f"Leiden better in {wins}/{len(by_gse)} cohorts")
    pl(ax, "a", x=-0.5)

    ax2 = axes[1]
    for a in ["Leiden γ=1.0", "WGCNA-style Hierarchical"]:
        v = piv[a].values
        ax2.hist(v, bins=14, alpha=0.6, color=C(a), label=a, edgecolor="white",
                 linewidth=0.3)
    ax2.axvline(0, ls=":", color="#999999", lw=0.7)
    ax2.set_xlabel("Functional coherence $z$")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=5.4)
    pl(ax2, "b")
    o = OUT / "Fig4_consistency.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    return o


if __name__ == "__main__":
    df = load()
    print("✅", fig3(df).name)
    print("✅", fig4(df).name)
