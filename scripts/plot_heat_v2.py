#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 3 & 4 (期刊级重绘): 性能画像热图 + 跨队列一致性"""
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
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
OUT = BASE / "output/core_results"
FIG = BASE / "output/figures_v3"
JS.apply_style()


def fig3(df):
    """性能画像热图（按相似度排序，两段色阶）"""
    metrics = {"coh_z_mean": "Coherence $z$", "modularity_Q": "Modularity $Q$",
               "stability_ARI": "Stability", "ARI_noise0.1": "Robustness",
               "antag_split_rate": "Antagonistic", "n_modules": "Modules"}
    agg = df.groupby("algo")[list(metrics)].mean()
    agg = agg.dropna(subset=["coh_z_mean"])
    # z-score 标准化（按列），缺失值用列均值填补
    Z = (agg - agg.mean()) / (agg.std() + 1e-9)
    Z = Z.fillna(0.0)
    # 按行聚类排序
    order = leaves_list(linkage(pdist(Z.values, metric="euclidean"),
                                method="average"))
    Z = Z.iloc[order]
    M = agg.iloc[order]

    fig = plt.figure(figsize=(JS.DOUBLE, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.06,
                          left=0.135, right=0.90, top=0.90, bottom=0.07)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(Z.values, aspect="auto", cmap=JS.DIV_RB,
                   norm=TwoSlopeNorm(vmin=-2.2, vcenter=0, vmax=2.2))
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([JS.short_algo(m) if False else v
                        for v in metrics.values()], rotation=38,
                       ha="right", fontsize=5.0, rotation_mode="anchor")
    ax.set_yticks(range(len(Z)))
    ax.set_yticklabels([JS.short_algo(n) for n in Z.index], fontsize=4.6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(Z)):
        nm = Z.index[i]
        if str(nm).startswith("FACEO") or str(nm).startswith("Leiden γ=1.0") \
                or str(nm).startswith("WGCNA"):
            ax.text(-0.35, i, "▪", color=JS.METHOD_COLORS.get(nm, "#333333"),
                    fontsize=5, ha="right", va="center",
                    transform=ax.get_yaxis_transform())
    JS.panel_label(ax, "a", x=-0.30, y=1.035)

    cax = fig.add_axes([0.915, 0.30, 0.009, 0.42])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$z$-score (column-wise)", fontsize=5.4)
    cb.ax.tick_params(labelsize=4.6, width=0.4, length=1.2)
    cb.outline.set_linewidth(0.4)

    # 右：原始值热图（关键指标）
    ax2 = fig.add_subplot(gs[0, 1])
    key = ["coh_z_mean", "antag_split_rate", "ARI_noise0.1", "stability_ARI"]
    V = M[key]
    Vn = (V - V.min()) / (V.max() - V.min() + 1e-9)
    im2 = ax2.imshow(Vn.values, aspect="auto", cmap=JS.SEQ_BLUE)
    ax2.set_xticks(range(len(key)))
    ax2.set_xticklabels([JS.short_algo(m) if False else metrics[k]
                         for k in key], rotation=38, ha="right",
                        fontsize=5.0, rotation_mode="anchor")
    ax2.set_yticks(range(len(V)))
    ax2.set_yticklabels([JS.short_algo(n) for n in V.index], fontsize=4.6)
    ax2.tick_params(length=0)
    for s in ax2.spines.values():
        s.set_visible(False)
    for i, nm in enumerate(V.index):
        if str(nm).startswith(("FACEO", "Leiden γ=1.0", "WGCNA")):
            ax2.text(-0.30, i, "▪", color=JS.METHOD_COLORS.get(nm, "#333333"),
                     fontsize=5, ha="right", va="center",
                     transform=ax2.get_yaxis_transform())
    JS.panel_label(ax2, "b", x=-0.26, y=1.035)

    cax2 = fig.add_axes([0.945, 0.30, 0.009, 0.42])
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label("min–max scaled", fontsize=5.4)
    cb2.ax.tick_params(labelsize=4.6, width=0.4, length=1.2)
    cb2.outline.set_linewidth(0.4)

    o = FIG / "Fig3_heatmap.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")


def fig4(df):
    """跨队列一致性"""
    lg = df.copy()
    lg["algo"] = lg.algo.astype(str)
    rows = []
    for (gse, pw), sub in lg.groupby(["gse", "pathway"]):
        d = sub.set_index("algo")["coh_z_mean"]
        if "Leiden γ=1.0" in d.index and "WGCNA-style Hierarchical" in d.index:
            le = float(d.loc["Leiden γ=1.0"])
            wg = float(d.loc["WGCNA-style Hierarchical"])
            if np.isfinite(le) and np.isfinite(wg):
                rows.append({"gse": gse, "delta": le - wg,
                             "leiden": le, "wgcna": wg})
    r = pd.DataFrame(rows)
    if len(r) == 0:
        print("⚠️ Fig4: 无可用配对，跳过")
        return

    fig = plt.figure(figsize=(JS.ONE_HALF, 2.5))
    gs = fig.add_gridspec(1, 2, wspace=0.45, left=0.115, right=0.985,
                          top=0.86, bottom=0.16)

    # (a) 配对斜率
    ax = fig.add_subplot(gs[0, 0])
    for _, row in r.iterrows():
        col = JS.C_OURS if row["delta"] > 0 else JS.C_WGCNA
        ax.plot([0, 1], [row["wgcna"], row["leiden"]], color=col, lw=0.5,
                alpha=0.45, zorder=2)
    ax.scatter([0] * len(r), r["wgcna"], color=JS.C_WGCNA, s=8, zorder=4,
               edgecolors="white", linewidths=0.3)
    ax.scatter([1] * len(r), r["leiden"], color=JS.C_OURS, s=8, zorder=4,
               edgecolors="white", linewidths=0.3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WGCNA", "Leiden\n$\\gamma$=1.0"],
                                              fontsize=5.4)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("Coherence $z$ (per pathway)", fontsize=6)
    ax.set_title("Paired comparison", fontsize=6.5, pad=3)
    win = (r.delta > 0).sum()
    ax.text(0.5, 0.04, f"Leiden better in {win}/{len(r)}",
            transform=ax.transAxes, fontsize=5.0, ha="center", color="#444444")
    JS.tidy_axis(ax, grid_axis="y"); JS.panel_label(ax, "a", x=-0.30)

    # (b) Δ 分布
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(r.delta, bins=14, color="#B8C4D9", edgecolor="white",
            linewidth=0.5, zorder=3)
    ax.axvline(0, color="#333333", lw=0.7, zorder=4)
    ax.axvline(r.delta.median(), color=JS.C_OURS, lw=0.9, ls="--", zorder=4)
    ax.set_xlabel("$\\Delta$ coherence (Leiden − WGCNA)", fontsize=6)
    ax.set_ylabel("Pathway–cohort pairs", fontsize=6)
    ax.set_title("Distribution of gain", fontsize=6.5, pad=3)
    ax.text(0.96, 0.93, f"median = {r.delta.median():+.2f}",
            transform=ax.transAxes, fontsize=5.0, ha="right", va="top",
            color=JS.C_OURS)
    JS.tidy_axis(ax, grid_axis="y"); JS.panel_label(ax, "b", x=-0.26)

    o = FIG / "Fig4_consistency.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")


def main():
    df = pd.read_csv(OUT / "full_eval_v3.csv")
    fig3(df); fig4(df)


if __name__ == "__main__":
    main()
