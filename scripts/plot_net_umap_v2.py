#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 8 & 9 (期刊级重绘): 网络结构 + 自监督嵌入 UMAP"""
import sys
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score as ari

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import journal_style as JS
import gwas_full_analysis as GA
import gwas_full_analysis_v3 as V3
import gwas_algorithms_v2 as A2
import gwas_selfsup as SS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
FIG = BASE / "output/figures_v3"
JS.apply_style()
PAL = JS.NPG


def load_subnet(gse="GSE137342", go="GO:0042981", maxg=110):
    raw = pd.read_csv(BASE / f"qcdata/{gse}.csv", index_col=0, quoting=3,
                      low_memory=False)
    raw.index = [str(i).strip().strip('"').upper() for i in raw.index]
    keep = [re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(i)) is not None
            for i in raw.index]
    df = V3.preprocess_robust(raw[keep])
    go_all = V3.load_go_sets()
    paths = dict(V3.select_pathways(go_all, set(df.index)))
    key = next(k for k in paths if go in k)
    genes = [g for g in paths[key] if g in df.index]
    v = df.loc[genes].var(axis=1).sort_values(ascending=False)
    genes = list(v.index[:maxg])
    tom = GA.compute_tom(df.loc[genes])
    g = GA.to_ig(tom, genes)
    return tom, g, genes, key


def draw_net(ax, tom, labels, seed=0):
    n = len(labels)
    G = nx.Graph()
    thr = np.percentile(tom[np.triu_indices(n, 1)], 92)
    for i in range(n):
        for j in range(i + 1, n):
            if tom[i, j] >= thr:
                G.add_edge(i, j, weight=float(tom[i, j]))
    if G.number_of_edges() < 5:
        return
    pos = nx.spring_layout(G, seed=seed, k=0.45, iterations=80, weight="weight")
    uniq = sorted(set(labels))
    cmap = {u: PAL[i % len(PAL)] for i, u in enumerate(uniq)}
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.18, width=0.28,
                           edge_color="#AAAAAA")
    deg = dict(G.degree())
    mx = max(deg.values()) if deg else 1
    sizes = [6 + 34 * deg[i] / mx for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=[cmap[labels[i]] for i in G.nodes()],
                           node_size=sizes, linewidths=0.22,
                           edgecolors="white")
    ax.axis("off")


def fig8(tom, g, genes):
    fn = dict(GA.ALGOS); fn.update(A2.NEW_ALGOS)
    items = [("WGCNA-style\nHierarchical", "WGCNA-style"),
             ("Leiden γ=1.0", "Leiden  $\\gamma$=1.0"),
             ("Leiden γ=2.0", "Leiden  $\\gamma$=2.0")]
    fig, axes = plt.subplots(1, 3, figsize=(JS.DOUBLE, 2.35))
    fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.86, bottom=0.02)
    for ax, (an, title) in zip(axes, items):
        lab = np.asarray(fn[an](g))
        draw_net(ax, tom, lab)
        ax.set_title(f"{title}\n{len(set(lab))} modules", fontsize=6.6, pad=3)
    for ax, L in zip(axes, "abc"):
        JS.panel_label(ax, L, x=0.02, y=0.99, size=8)
    o = FIG / "Fig8_network_structure.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")


def fig9(tom, g, genes):
    x, ei, ea = SS.build_pyg_data(tom)
    emb, losses = SS.train_edge_prediction(x, ei, ea, encoder="gat", epochs=300)
    emb = np.asarray(emb)
    lab_emb = SS.embed_to_partition(emb)
    fn = dict(GA.ALGOS); fn.update(A2.NEW_ALGOS)
    lab_w = np.asarray(fn["WGCNA-style\nHierarchical"](g))
    lab_l = np.asarray(fn["Leiden γ=1.0"](g))
    lab_e = np.asarray(fn["Leiden γ=2.0"](g))
    try:
        import umap
        proj = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3,
                         random_state=42).fit_transform(emb)
    except Exception:
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=15, random_state=42,
                    init="pca").fit_transform(emb)

    fig = plt.figure(figsize=(JS.DOUBLE, 2.3))
    gs = fig.add_gridspec(1, 4, wspace=0.10, left=0.012, right=0.99,
                          top=0.86, bottom=0.03)
    panels = [(lab_w, "WGCNA-style", "a"), (lab_l, "Leiden  $\\gamma$=1.0", "b"),
              (lab_e, "Leiden  $\\gamma$=2.0", "c")]
    for i, (lab, title, L) in zip(range(3), panels):
        ax = fig.add_subplot(gs[0, i])
        uniq = sorted(set(lab))
        cmap = {u: PAL[j % len(PAL)] for j, u in enumerate(uniq)}
        ax.scatter(proj[:, 0], proj[:, 1], c=[cmap[v] for v in lab], s=7,
                   linewidths=0.2, edgecolors="white", alpha=0.95)
        a = ari(lab, lab_emb) if len(set(lab_emb)) > 1 else np.nan
        ax.set_title(f"{title}\n{len(uniq)} mod.  ARI={a:.3f}", fontsize=6.2, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        JS.panel_label(ax, L, x=0.02, y=0.99)

    ax = fig.add_subplot(gs[0, 3])
    vals = [ari(lab_w, lab_emb), ari(lab_l, lab_emb), ari(lab_e, lab_emb)]
    names = ["WGCNA", "Leiden\n$\\gamma$=1.0", "Leiden\n$\\gamma$=2.0"]
    cols = [JS.C_WGCNA, JS.C_OURS, JS.NPG[4]]
    bars = ax.bar(range(3), vals, color=cols, width=0.6, edgecolor="white",
                  linewidth=0.5, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=5.0)
    ax.set_ylabel("ARI vs embedding", fontsize=6)
    ax.set_ylim(0, max(vals) * 1.30)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.03, f"{v:.2f}", ha="center", fontsize=5.0)
    ax.set_title("Independent agreement", fontsize=6.2, pad=3)
    JS.tidy_axis(ax, grid_axis="y"); JS.panel_label(ax, "d", x=-0.24)
    o = FIG / "Fig9_umap_embedding.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")


def main():
    tom, g, genes, key = load_subnet()
    print(f"通路: {key[:46]} | {len(genes)} 基因")
    fig8(tom, g, genes)
    fig9(tom, g, genes)


if __name__ == "__main__":
    main()
