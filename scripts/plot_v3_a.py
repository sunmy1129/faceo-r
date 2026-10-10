#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SCI 风格图表 v3（31 算法 × 11 数据集）"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

warnings.filterwarnings("ignore")

RES = Path("/Users/sunmy/Documents/QF/GWAS/output/core_results/full_eval_v3.csv")
OUT = Path("/Users/sunmy/Documents/QF/GWAS/output/figures_v3")
OUT.mkdir(parents=True, exist_ok=True)
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

HL = {
    "Leiden γ=0.5": "#F0E442", "Leiden γ=1.0": "#D55E00",
    "Leiden γ=2.0": "#E69F00", "Leiden γ=5.0": "#CC79A7",
    "Louvain": "#0072B2", "Infomap": "#009E73",
    "WGCNA-style Hierarchical": "#333333", "CPM γ=0.10": "#56B4E9",
    "HDBSCAN": "#8B4513", "Fast Greedy": "#9370DB",
}
DEF = "#CCCCCC"


def C(a):
    return HL.get(a, DEF)


def pl(ax, s, x=-0.16, y=1.05):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def load():
    df = pd.read_csv(RES)
    df["algo"] = df["algo"].astype(str)
    return df


def agg(df):
    cols = [c for c in ["n_modules", "modularity_Q", "coh_z_mean", "coh_mean",
                        "stability_ARI", "ARI_noise0.1", "ARI_noise0.2",
                        "ARI_edge0.1", "antag_split_rate", "ARI_vs_random"]
            if c in df.columns]
    return df.groupby("algo")[cols].mean()
