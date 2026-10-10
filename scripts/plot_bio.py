#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生物学验证图：模块 -> GO 术语对应"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import journal_style as JS

JS.apply_style()
R = Path("/Users/sunmy/Documents/QF/GWAS/output/core_results")
O = Path("/Users/sunmy/Documents/QF/GWAS/output/figures_v3")

d = pd.read_csv(R / "bio_validation.csv")
d["nlp"] = -np.log10(d["pval"].clip(lower=1e-300))
d["short"] = d["top_GO"].str.replace(r"\s*\(GO:\d+\)", "", regex=True).str[:38]
agg = d.groupby("short").agg(nlp=("nlp", "median"), n=("n", "median"),
                             k=("k", "median")).sort_values("nlp", ascending=True).tail(8)

fig, ax = plt.subplots(figsize=(JS.DOUBLE, 2.2))
y = np.arange(len(agg))
ax.barh(y, agg["nlp"].values, color=JS.C_OURS, height=0.62, edgecolor="white", linewidth=0.4)
ax.set_yticks(y)
ax.set_yticklabels([s[:34] for s in agg.index], fontsize=7)
ax.set_xlabel("$-\\log_{10}$ enrichment $p$ (median over modules)")
ax.set_title("Modules recovered by FACEO-R map onto coherent biological processes", fontsize=8.5)
for i, (v, k) in enumerate(zip(agg["nlp"].values, agg["k"].values)):
    ax.text(v + 1.5, i, f"{int(k)} genes", va="center", fontsize=6.5)
JS.tidy_axis(ax)
fig.tight_layout()
fig.savefig(O / "FigCA_bio.pdf")
fig.savefig(O / "FigCA_bio.png", dpi=600)
plt.close(fig)
print(f"✅ FigCA_bio  ({len(agg)} 个术语, 中位 -log10p={agg['nlp'].median():.0f})")
