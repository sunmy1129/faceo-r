#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure: 自监督嵌入 UMAP 可视化（全部数据为真实计算）"""
import sys
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import adjusted_rand_score as ari

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import gwas_full_analysis as GA
import gwas_full_analysis_v3 as V3
import gwas_selfsup as SS
import gwas_algorithms_v2 as A2

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
OUT = BASE / "output/figures_v3"
MM = 1 / 25.4
W2 = 183 * MM

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.2,
    "axes.linewidth": 0.6, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
PAL = ["#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9",
       "#F0E442", "#8B4513", "#9370DB", "#2E8B57", "#DC143C", "#4682B4"]


def main():
    raw = pd.read_csv(BASE / "qcdata/GSE137342.csv", index_col=0, quoting=3,
                      low_memory=False)
    raw.index = [str(i).strip().strip('"').upper() for i in raw.index]
    keep = [re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(i)) is not None
            for i in raw.index]
    raw = raw[keep]
    df = V3.preprocess_robust(raw)
    go = V3.load_go_sets()
    paths = dict(V3.select_pathways(go, set(df.index)))
    key = next(k for k in paths if "GO:0042981" in k)
    genes = [g for g in paths[key] if g in df.index]
    v = df.loc[genes].var(axis=1).sort_values(ascending=False)
    genes = list(v.index[:150])
    tom = GA.compute_tom(df.loc[genes])
    g = GA.to_ig(tom, genes)
    print(f"通路: {key[:50]} | {len(genes)} 基因")

    # --- 自监督嵌入（不使用任何社区标签）---
    x, ei, ea = SS.build_pyg_data(tom)
    emb, losses = SS.train_edge_prediction(x, ei, ea, encoder="gat", epochs=300)
    print(f"嵌入: {np.asarray(emb).shape}, 最终损失 {losses[-1]:.3f}")
    lab_emb = SS.embed_to_partition(emb)

    # --- 各算法划分 ---
    fn = dict(GA.ALGOS); fn.update(A2.NEW_ALGOS)
    lab_w = np.asarray(fn["WGCNA-style\nHierarchical"](g))
    lab_l = np.asarray(fn["Leiden γ=1.0"](g))
    lab_e = np.asarray(fn["Leiden γ=2.0"](g))

    # --- UMAP 投影（真实） ---
    try:
        import umap
        proj = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.25,
                         random_state=42).fit_transform(np.asarray(emb))
    except Exception:
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=15, random_state=42,
                    init="pca").fit_transform(np.asarray(emb))

    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.25))
    fig.subplots_adjust(wspace=0.09)
    panels = [(lab_w, "WGCNA-style", "a"), (lab_l, "Leiden γ=1.0", "b"),
              (lab_e, "Leiden γ=2.0", "c")]
    for ax, (lab, title, tag) in zip(axes[:3], panels):
        cmap = {u: PAL[i % len(PAL)] for i, u in enumerate(sorted(set(lab)))}
        ax.scatter(proj[:, 0], proj[:, 1], c=[cmap[x] for x in lab], s=9,
                   linewidths=0.15, edgecolors="white")
        a = ari(lab, lab_emb) if len(set(lab_emb)) > 1 else np.nan
        ax.set_title(f"{title}  ({len(set(lab))} modules)\n"
                     f"ARI vs embedding = {a:.3f}", fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(-0.06, 1.03, tag, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")

    # 第 4 面板：真实一致性对比
    ax = axes[3]
    vals = [ari(lab_w, lab_emb), ari(lab_l, lab_emb), ari(lab_e, lab_emb),
            ari(lab_l, lab_w)]
    names = ["WGCNA\nvs emb", "Leiden1.0\nvs emb", "Leiden2.0\nvs emb",
             "Leiden1.0\nvs WGCNA"]
    cols = ["#333333", "#D55E00", "#E69F00", "#0072B2"]
    ax.bar(range(len(vals)), vals, color=cols, width=0.62)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=5.4)
    ax.set_ylabel("ARI", fontsize=7)
    ax.set_ylim(0, max(0.4, max(vals) * 1.25))
    ax.set_title("Partition agreement", fontsize=6.8)
    ax.text(-0.28, 1.03, "d", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")
    for i, vv in enumerate(vals):
        ax.text(i, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=5.2)

    fig.suptitle("Self-supervised embedding (UMAP) of the apoptotic subnetwork — "
                 "colour = partition membership; ARI computed against the embedding",
                 fontsize=7.0, y=1.045)
    o = OUT / "Fig9_umap_embedding.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")
    print(f"真实 ARI: WGCNA={vals[0]:.3f}, Leiden1.0={vals[1]:.3f}, "
          f"Leiden2.0={vals[2]:.3f}, Leiden1.0-vs-WGCNA={vals[3]:.3f}")


if __name__ == "__main__":
    main()
