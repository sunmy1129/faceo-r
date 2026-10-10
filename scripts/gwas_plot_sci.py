#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCI 风格图表生成器（Nature/Science 期刊规范）
- 尺寸：单栏 89 mm / 双栏 183 mm
- 字体：Arial 7-9 pt（期刊正文尺度）
- 分辨率：矢量 PDF + 600 dpi TIFF/PNG
- 配色：色盲友好调色板
- 面板：小写粗体标注 (a, b, c)
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

RES = Path("/Users/sunmy/Documents/QF/GWAS/output/core_results/full_eval.csv")
OUTDIR = Path("/Users/sunmy/Documents/QF/GWAS/output/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4  # mm -> inch
W1, W2 = 89 * MM, 183 * MM   # 单栏 / 双栏宽度

# ---------------- SCI 风格配置 ----------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "legend.fontsize": 6.3,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,      # TrueType，期刊要求可编辑文本
    "ps.fonttype": 42,
})

# 色盲友好调色板（Okabe-Ito 扩展 + Nature 常用）
PAL = {
    "hier": "#7F7F7F",
    "leiden": "#D55E00",
    "cpm": "#E69F00",
    "louvain": "#0072B2",
    "other": "#56B4E9",
    "ml": "#009E73",
    "embed": "#CC79A7",
}
FAMILY_COLOR = {
    "classical": "#0072B2",
    "embedding": "#CC79A7",
}


def panel_label(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def load():
    df = pd.read_csv(RES)
    df["algo"] = df["algo"].astype(str)
    return df


def agg(df):
    """按算法汇总（跨数据集/通路）"""
    cols = ["n_modules", "modularity_Q", "coh_z_mean", "coh_mean", "purity_mean",
            "size_entropy", "unassigned_rate", "stability_ARI",
            "antag_split_rate", "ARI_noise0.1", "ARI_noise0.2",
            "ARI_edge0.1", "ARI_vs_random"]
    cols = [c for c in cols if c in df.columns]
    g = df.groupby("algo")[cols].agg(["mean", "std", "count"])
    return g


# ============================================================
# 图 1：过分割现象 —— 模块数 vs 模块度（核心发现）
# ============================================================
def fig1_overpartition(df):
    g = df.groupby("algo").agg(
        n_modules=("n_modules", "mean"), n_sd=("n_modules", "std"),
        Q=("modularity_Q", "mean"), Q_sd=("modularity_Q", "std"),
        purity=("coh_z_mean", "mean"), fam=("family", "first")).reset_index()
    g = g.sort_values("n_modules")

    fig = plt.figure(figsize=(W2, 2.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.9], wspace=0.42)

    # (a) 模块数 vs 模块度
    ax = fig.add_subplot(gs[0])
    for _, r in g.iterrows():
        c = FAMILY_COLOR.get(r["fam"], "#333333")
        ax.errorbar(r["n_modules"], r["Q"], xerr=r["n_sd"] or 0, yerr=r["Q_sd"] or 0,
                    fmt="o", ms=3.6, color=c, ecolor=c, elinewidth=0.6,
                    capsize=1.2, alpha=0.9, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Number of modules (log scale)")
    ax.set_ylabel("Modularity $Q$")
    ax.set_title("Fragmentation cost")
    # 标注关键点
    for name, short in [("Leiden γ=0.5", "γ=0.5"), ("Leiden γ=1.0", "γ=1.0"),
                        ("Leiden γ=2.0", "γ=2.0"), ("Leiden γ=5.0", "γ=5.0")]:
        row = g[g.algo == name]
        if len(row):
            r = row.iloc[0]
            ax.annotate(short, (r["n_modules"], r["Q"]), fontsize=5.6,
                        xytext=(3, 3), textcoords="offset points", color=PAL["leiden"])
    panel_label(ax, "a")

    # (b) 模块数 vs 功能纯度
    ax2 = fig.add_subplot(gs[1])
    for _, r in g.iterrows():
        c = FAMILY_COLOR.get(r["fam"], "#333333")
        ax2.scatter(r["n_modules"], r["purity"], s=14, color=c, alpha=0.9,
                    edgecolor="white", linewidth=0.3, zorder=3)
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of modules (log scale)")
    ax2.set_ylabel("Functional coherence $z$-score")
    ax2.set_title("More modules ≠ more accurate")
    panel_label(ax2, "b")

    # (c) 分辨率扫描曲线
    ax3 = fig.add_subplot(gs[2])
    lg = df[df.algo.str.startswith("Leiden")]
    if len(lg):
        piv = lg.groupby("algo").agg(n=("n_modules", "mean"), Q=("modularity_Q", "mean"),
                                     p=("coh_z_mean", "mean")).reset_index()
        piv["gamma"] = piv.algo.str.extract(r"γ=([0-9.]+)").astype(float)
        piv = piv.sort_values("gamma")
        ax3.plot(piv["gamma"], piv["Q"], "o-", color=PAL["leiden"], ms=3.5, lw=1.0,
                 label="Modularity $Q$")
        ax3.set_xlabel("Resolution parameter $\\gamma$")
        ax3.set_ylabel("Modularity $Q$", color=PAL["leiden"])
        ax3.tick_params(axis="y", labelcolor=PAL["leiden"])
        ax3.set_xscale("log")
        ax3.set_title("Resolution collapse")
        ax3b = ax3.twinx()
        ax3b.plot(piv["gamma"], piv["p"], "s--", color="#009E73", ms=3.2, lw=1.0,
                  label="Coherence $z$")
        ax3b.set_ylabel("Coherence $z$-score", color="#009E73")
        ax3b.tick_params(axis="y", labelcolor="#009E73")
        ax3b.spines["right"].set_visible(True)
        ax3b.spines["top"].set_visible(False)
        panel_label(ax3, "c")

    out = OUTDIR / "Fig1_overpartition.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    return out


# ============================================================
# 图 2：多指标综合对比（功能纯度 / 稳定性 / 鲁棒性 / 拮抗拆分）
# ============================================================
def fig2_metrics(df):
    g = df.groupby("algo").agg(
        purity=("coh_z_mean", "mean"), pure_sd=("coh_z_mean", "std"),
        stab=("stability_ARI", "mean"),
        rob=("ARI_noise0.1", "mean"),
        antag=("antag_split_rate", "mean"),
        nmod=("n_modules", "mean"),
        n=("n_modules", "count")).reset_index()
    g = g.sort_values("purity", ascending=False)
    # 只保留在全部数据集中都出现的算法
    g = g[g["n"] >= g["n"].max() * 0.9]

    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.35))
    fig.subplots_adjust(wspace=0.5)

    # (a) 功能纯度排名
    ax = axes[0]
    y = np.arange(len(g))
    ax.barh(y, g["purity"], xerr=g["pure_sd"], color="#009E73",
            height=0.68, error_kw=dict(lw=0.5, capsize=1))
    ax.set_yticks(y)
    ax.set_yticklabels(g["algo"], fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlabel("Functional coherence $z$-score")
    panel_label(ax, "a", x=-0.42)

    # (b) 稳定性 vs 鲁棒性
    ax2 = axes[1]
    for _, r in g.iterrows():
        ax2.scatter(r["stab"], r["rob"], s=16, color="#0072B2", alpha=0.85,
                    edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Stability (bootstrap ARI)")
    ax2.set_ylabel("Robustness (ARI under 10% noise)")
    lim = [min(g["stab"].min(), g["rob"].min()) - 0.05,
           max(g["stab"].max(), g["rob"].max()) + 0.05]
    ax2.plot(lim, lim, ":", color="#999999", lw=0.6, zorder=0)
    panel_label(ax2, "b")

    # (c) 拮抗拆分率（含模块数参考）
    ax3 = axes[2]
    g2 = g.sort_values("antag", ascending=False)
    x = np.arange(len(g2))
    ax3.bar(x, g2["antag"], color=PAL["leiden"], width=0.66)
    ax3.set_xticks(x)
    ax3.set_xticklabels(g2["algo"], rotation=90, fontsize=5.4)
    ax3.set_ylabel("Antagonistic pair\nseparation rate")
    ax3.set_ylim(0, 1.08)
    ax3.axhline(0.5, ls=":", color="#999999", lw=0.6)
    panel_label(ax3, "c", x=-0.2)

    out = OUTDIR / "Fig2_metrics.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    return out


# ============================================================
# 图 3：跨数据集一致性热图
# ============================================================
def fig3_heatmap(df):
    metrics = [c for c in ["n_modules", "modularity_Q", "coh_z_mean",
                           "stability_ARI", "antag_split_rate"] if c in df.columns]
    piv = df.groupby(["algo", "gse"])[metrics].mean()
    top = df.groupby("algo")["coh_z_mean"].mean().sort_values(ascending=False)
    algos = list(top.index)[:12]
    data, labels = [], []
    for a in algos:
        row = []
        for m in metrics:
            try:
                row.append(piv.loc[(a, slice(None)), m].mean())
            except Exception:
                row.append(np.nan)
        data.append(row)
        labels.append(a)
    arr = np.array(data, dtype=float)
    # 按列 z-score（便于跨指标比较）
    z = (arr - np.nanmean(arr, 0)) / (np.nanstd(arr, 0) + 1e-9)

    fig, ax = plt.subplots(figsize=(W1 * 1.35, 2.6))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-1.8, vmax=1.8, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=5.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.8,
                        color="black" if abs(z[i, j]) < 1.4 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("z-score", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_title("Algorithm performance profile (mean over 4 datasets × 7 pathways)",
                 fontsize=7)
    out = OUTDIR / "Fig3_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    return out


# ============================================================
# 图 4：自监督嵌入独立验证（对比 v1.0 监督式）
# ============================================================
def fig4_embedding(df):
    emb = df[df.family == "embedding"].copy()
    if not len(emb):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(W1 * 1.9, 2.4))
    fig.subplots_adjust(wspace=0.4)

    # (a) 嵌入划分的功能纯度 vs 传统算法
    ax = axes[0]
    clas = df[df.family == "classical"].groupby("algo")["coh_z_mean"].mean()
    e = emb.groupby("algo")["coh_z_mean"].mean()
    names = list(clas.index) + list(e.index)
    vals = list(clas.values) + list(e.values)
    cols = ["#9ecae1"] * len(clas) + [PAL["embed"]] * len(e)
    ax.barh(np.arange(len(names)), vals, color=cols, height=0.68)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlabel("Functional coherence $z$-score")
    panel_label(ax, "a", x=-0.5)

    # (b) 嵌入与 Leiden 的独立一致性 ARI
    ax2 = axes[1]
    if "ari_vs_leiden_g1.0" in emb.columns:
        v = emb["ari_vs_leiden_g1.0"].dropna().values
        ax2.hist(v, bins=12, color=PAL["embed"], alpha=0.85, edgecolor="white", linewidth=0.4)
        ax2.axvline(np.mean(v), color="black", ls="--", lw=0.8)
        ax2.text(np.mean(v), ax2.get_ylim()[1] * 0.92,
                 f"mean = {np.mean(v):.2f}", fontsize=6, ha="left")
        ax2.set_xlabel("ARI (self-supervised embedding vs Leiden γ=1.0)")
        ax2.set_ylabel("Count")
        ax2.set_title("Independent verification")
    panel_label(ax2, "b")
    out = OUTDIR / "Fig4_embedding.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    return out


def main():
    df = load()
    print(f"载入 {len(df)} 条记录 | 算法 {df.algo.nunique()} 个 | 组合 {df.groupby(['gse','pathway']).ngroups} 个")
    outs = []
    for fn in (fig1_overpartition, fig2_metrics, fig3_heatmap, fig4_embedding):
        try:
            o = fn(df)
            if o:
                outs.append(o)
                print("✅", o.name)
        except Exception as e:
            print(f"❌ {fn.__name__}: {str(e)[:90]}")
    return outs


if __name__ == "__main__":
    main()
