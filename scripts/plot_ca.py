#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""clean vs 降质 + alpha 消融图"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import journal_style as JS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/Users/sunmy/Documents/QF/GWAS/output/core_results")
FIG = Path("/Users/sunmy/Documents/QF/GWAS/output/figures_v3")
JS.apply_style()

d = pd.read_csv(R / "faceo_r_enhanced.csv")
piv = d.pivot_table(index="condition", columns="algo", values="coh", aggfunc="mean")
ORDER = ["clean_0.0", "dropout_0.5", "noise_0.5", "noise_1.0", "samples_0.15", "samples_0.3"]
LBL = ["Clean", "Dropout\n50%", "Noise\n0.5", "Noise\n1.0", "Samples\n15%", "Samples\n30%"]
piv = piv.reindex([c for c in ORDER if c in piv.index])

fig, ax = plt.subplots(figsize=(3.45, 2.5))
x = np.arange(len(piv))
w = 0.26
cols = [("WGCNA-style", JS.C_WGCNA), ("Leiden γ=1.0", JS.C_LEIDEN), ("FACEO-R (ours)", JS.C_OURS)]
for i, (m, c) in enumerate(cols):
    if m in piv.columns:
        ax.bar(x + (i - 1) * w, piv[m].values, w, color=c, label=JS.short_algo(m),
               edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(LBL, fontsize=6.5)
ax.set_ylabel("Functional coherence (median $z$)", fontsize=7.5)
ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, 1.16), columnspacing=1.0, handlelength=1.1)
JS.tidy_axis(ax)
fig.tight_layout()
fig.savefig(FIG / "FigCA_clean.pdf")
fig.savefig(FIG / "FigCA_clean.png", dpi=600)
plt.close(fig)
print("✅ FigCA_clean")

# ---- alpha 扫描/消融 ----
am = {c: piv.loc[c].get("FACEO-R (ours)", np.nan) for c in piv.index} if False else None
alphas, means = [], []
for col in [c for c in d.algo.unique() if c.startswith("alpha=")]:
    alphas.append(float(col.split("=")[1]))
    means.append(d[d.algo == col].coh.mean())
o = np.argsort(alphas)
alphas = np.array(alphas)[o]; means = np.array(means)[o]

fig, ax = plt.subplots(figsize=(3.45, 2.4))
ax.plot(alphas, means, "o-", color=JS.C_OURS, ms=4, lw=1.4, label="Fixed $\\alpha$")
ad = d[d.algo == "FACEO-R (ours)"].coh.mean()
nw = d[d.algo == "FACEO-R-narrow"].coh.mean() if "FACEO-R-narrow" in set(d.algo) else np.nan
ax.axhline(ad, color=JS.C_LEIDEN, ls="--", lw=1.2, label="Original adaptive")
if not np.isnan(nw):
    ax.axhline(nw, color=JS.C_CPM, ls=":", lw=1.4, label="Retuned adaptive")
ax.set_xlabel("Anchoring weight $\\alpha$", fontsize=7.5)
ax.set_ylabel("Mean functional coherence", fontsize=7.5)
ax.legend(frameon=False, fontsize=6, loc="lower left")
JS.tidy_axis(ax)
fig.tight_layout()
fig.savefig(FIG / "FigCA_alpha.pdf")
fig.savefig(FIG / "FigCA_alpha.png", dpi=600)
plt.close(fig)
print("✅ FigCA_alpha")
