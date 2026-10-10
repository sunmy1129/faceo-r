#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充 SCI 图表：合成基准 / 零模型 / 下游任务"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
RES = BASE / "output/core_results"
FIG = BASE / "output/figures"
FIG.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
W1, W2 = 89 * MM, 183 * MM

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "legend.fontsize": 6.0, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})

HILITE = {"Leiden γ=1.0": "#D55E00", "Leiden γ=2.0": "#E69F00",
          "Leiden γ=5.0": "#F0E442", "Louvain": "#0072B2",
          "WGCNA-style Hierarchical": "#333333", "CPM γ=0.10": "#CC79A7",
          "Infomap": "#009E73", "Spectral": "#56B4E9"}
DEFAULT_C = "#BBBBBB"


def pl(ax, s, x=-0.15, y=1.05):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def c_of(name):
    return HILITE.get(name, DEFAULT_C)


def fig5_synthetic():
    f = RES / "synthetic_results.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df["algo"] = df["algo"].astype(str)
    fig = plt.figure(figsize=(W2, 2.4))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1, 1, 1])

    # (a) LFR 梯度
    ax = fig.add_subplot(gs[0])
    lfr = df[df.benchmark == "LFR"]
    piv = lfr.groupby(["algo", "mu"])["ARI"].mean().unstack()
    highlight = ["Leiden γ=1.0", "Louvain", "WGCNA-style Hierarchical",
                 "Leiden γ=5.0", "CPM γ=0.05", "Infomap"]
    for a in piv.index:
        if a in highlight:
            ax.plot(piv.columns, piv.loc[a], "o-", ms=3, lw=1.0, color=c_of(a), label=a, zorder=3)
        else:
            ax.plot(piv.columns, piv.loc[a], "-", lw=0.5, color=DEFAULT_C, alpha=0.5, zorder=1)
    ax.set_xlabel("Community fuzziness $\\mu$")
    ax.set_ylabel("ARI (vs ground truth)")
    ax.set_title("Synthetic block model")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=5.2, loc="lower left", ncol=1)
    pl(ax, "a")

    # (b) 嵌套子群
    ax2 = fig.add_subplot(gs[1])
    sub = df[df.benchmark == "Nested_sub"].groupby("algo").agg(
        ARI_sub=("ARI", "mean"), n=("n_modules", "mean"))
    mod = df[df.benchmark == "Nested_mod"].groupby("algo")["ARI"].mean()
    j = sub.join(mod.rename("ARI_mod")).sort_values("ARI_sub")
    y = np.arange(len(j))
    ax2.barh(y, j["ARI_sub"], color=[c_of(a) for a in j.index], height=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels(j.index, fontsize=5.4)
    ax2.set_xlabel("ARI vs nested sub-groups")
    ax2.set_xlim(0, 1.08)
    ax2.axvline(j["ARI_sub"].max(), ls=":", color="#999999", lw=0.6)
    ax2.set_title("Nested sub-structure")
    pl(ax2, "b", x=-0.55)

    # (c) 噪声梯度
    ax3 = fig.add_subplot(gs[2])
    nz = df[df.benchmark == "Noise"].groupby(["algo", "noise"])["ARI"].mean().unstack()
    for a in nz.index:
        if a in highlight:
            ax3.plot(nz.columns, nz.loc[a], "o-", ms=3, lw=1.0, color=c_of(a), zorder=3)
        else:
            ax3.plot(nz.columns, nz.loc[a], "-", lw=0.5, color=DEFAULT_C, alpha=0.5, zorder=1)
    ax3.set_xlabel("Noise level $\\sigma$")
    ax3.set_ylabel("ARI")
    ax3.set_title("Noise robustness")
    ax3.set_ylim(-0.05, 1.05)
    pl(ax3, "c", x=-0.15)

    out = FIG / "Fig5_synthetic.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    return out


def fig6_null():
    f = RES / "null_models.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(W1 * 1.9, 2.3))
    fig.subplots_adjust(wspace=0.38)

    # (a) 真实 vs 零分布
    ax = axes[0]
    x = np.arange(len(df))
    ax.scatter(x, df["Q_real"], s=10, color="#D55E00", label="Observed $Q$", zorder=3)
    ax.scatter(x, df["Q_nullA_mean"], s=10, color="#0072B2", marker="s",
               label="Null A (degree-rewired)", zorder=3)
    ax.scatter(x, df["Q_nullB_mean"], s=10, color="#009E73", marker="^",
               label="Null B (expression-permuted)", zorder=3)
    ax.set_xlabel("Dataset × pathway combination (sorted)")
    ax.set_ylabel("Modularity $Q$")
    ax.legend(fontsize=5.4, loc="upper left")
    pl(ax, "a")

    # (b) z 分数分布
    ax2 = axes[1]
    ax2.hist(df["z_nullA"].dropna(), bins=12, color="#0072B2", alpha=0.75,
             edgecolor="white", linewidth=0.4, label="Null A")
    ax2.hist(df["z_nullB"].dropna(), bins=12, color="#009E73", alpha=0.55,
             edgecolor="white", linewidth=0.4, label="Null B")
    ax2.axvline(2, ls="--", color="black", lw=0.7)
    ax2.text(2.2, ax2.get_ylim()[1] * 0.85, "$z=2$", fontsize=5.6)
    ax2.set_xlabel("$z$-score vs null model")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=5.6)
    ax2.set_title("Structure is not random")
    pl(ax2, "b")
    out = FIG / "Fig6_null.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    return out


def fig7_downstream():
    f = RES / "downstream_results.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if not len(df):
        return None
    df["algo"] = df["algo"].astype(str)
    tasks = df["task"].unique()
    fig, axes = plt.subplots(1, min(2, len(tasks)) + 1, figsize=(W2, 2.3))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    fig.subplots_adjust(wspace=0.5)

    # (a) 各任务的算法 AUC
    ax = axes[0]
    piv = df.groupby(["algo", "task"])["AUC"].mean().unstack()
    piv = piv.sort_values(piv.columns[0], ascending=False)
    x = np.arange(len(piv))
    ncol = len(piv.columns)
    w = 0.8 / ncol
    cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for i, c in enumerate(piv.columns):
        ax.bar(x + i * w, piv[c], width=w, color=cols[i % 4], label=c)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(piv.index, rotation=90, fontsize=5.0)
    ax.set_ylabel("CV AUC")
    ax.legend(fontsize=5.2, ncol=2)
    ax.set_title("Downstream prediction")
    pl(ax, "a", x=-0.2)

    # (b) AUC vs 模块数
    if len(axes) > 1:
        ax2 = axes[1]
        mrg = df.merge(
            pd.read_csv(RES / "full_eval.csv").groupby(["gse", "pathway", "algo"])
            [["n_modules"]].mean().reset_index(),
            on=["gse", "pathway", "algo"], how="left")
        for t in tasks:
            s = mrg[mrg.task == t]
            ax2.scatter(s["n_modules"], s["AUC"], s=8, alpha=0.6, label=t)
        ax2.set_xscale("log")
        ax2.set_xlabel("Number of modules (log)")
        ax2.set_ylabel("CV AUC")
        ax2.legend(fontsize=5.4)
        pl(ax2, "b")
    out = FIG / "Fig7_downstream.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for fn in (fig5_synthetic, fig6_null, fig7_downstream):
        try:
            o = fn()
            print(("✅ " + o.name) if o else f"⏸ {fn.__name__}: 数据未就绪")
        except Exception as e:
            print(f"❌ {fn.__name__}: {str(e)[:100]}")
