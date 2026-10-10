#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure: 真实通路网络结构对比（WGCNA vs Leiden 多分辨率）"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import gwas_full_analysis as GA
import gwas_full_analysis_v3 as V3
import gwas_algorithms_v2 as A2

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
OUT = BASE / "output/figures_v3"
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
W2 = 183 * MM

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "axes.linewidth": 0.6, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

PAL = ["#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9",
       "#F0E442", "#8B4513", "#9370DB", "#2E8B57", "#DC143C", "#4682B4"]


def build(gse, pw_key, max_genes=110):
    raw = pd.read_csv(BASE / f"qcdata/{gse}.csv", index_col=0, quoting=3,
                      low_memory=False)
    raw.index = [str(i).strip().strip('"').upper() for i in raw.index]
    import re
    keep = [re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(i)) is not None for i in raw.index]
    raw = raw[keep]
    df = V3.preprocess_robust(raw)
    go = V3.load_go_sets()
    paths = dict(V3.select_pathways(go, set(df.index)))
    key = next((k for k in paths if pw_key in k), None)
    if key is None:
        raise KeyError(f"{pw_key} not found; available: {list(paths)[:3]}")
    genes = [g for g in paths[key] if g in df.index]
    v = df.loc[genes].var(axis=1).sort_values(ascending=False)
    genes = list(v.index[:max_genes])
    tom = GA.compute_tom(df.loc[genes])
    g = GA.to_ig(tom, genes)
    return tom, g, genes


def draw(ax, tom, labels, genes, title, seed=0):
    G = nx.Graph()
    n = len(genes)
    thr = np.percentile(tom[np.triu_indices(n, 1)], 92)
    for i in range(n):
        for j in range(i + 1, n):
            if tom[i, j] >= thr:
                G.add_edge(i, j, weight=float(tom[i, j]))
    if G.number_of_edges() < 5:
        ax.text(0.5, 0.5, "sparse", ha="center", transform=ax.transAxes)
        return
    pos = nx.spring_layout(G, seed=seed, k=0.4, iterations=60,
                           weight="weight")
    uniq = sorted(set(labels))
    cmap = {u: PAL[i % len(PAL)] for i, u in enumerate(uniq)}
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.22, width=0.35,
                           edge_color="#888888")
    nc = [cmap[labels[i]] for i in G.nodes()]
    sizes = [18 + 90 * G.degree(i) / max(1, max(dict(G.degree()).values()))
             for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=sizes,
                           linewidths=0.25, edgecolors="white")
    ax.set_title(title, fontsize=7.6)
    ax.axis("off")


def main():
    tom, g, genes = build("GSE137342", "GO:0042981", max_genes=110)
    algos = [("WGCNA-style\nHierarchical", "WGCNA-style\nhierarchical cutting"),
             ("Leiden γ=1.0", "Leiden  γ = 1.0"),
             ("Leiden γ=2.0", "Leiden  γ = 2.0")]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.5))
    fig.subplots_adjust(wspace=0.05)
    for ax, (aname, title) in zip(axes, algos):
        fn = dict(GA.ALGOS)
        fn.update(A2.NEW_ALGOS)
        lab = np.asarray(fn[aname](g))
        k = len(set(lab))
        draw(ax, tom, lab, genes, f"{title}\n{k} modules")
    fig.suptitle("Apoptotic process subnetwork (GSE137342, 110 genes, shared TOM input)",
                 fontsize=7.8, y=1.02)
    o = OUT / "Fig8_network_structure.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")
    return o


if __name__ == "__main__":
    main()
