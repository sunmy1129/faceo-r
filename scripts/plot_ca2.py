#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重绘：clean 对比 + alpha 扫描（双栏宽，SCI 风格）"""
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import journal_style as JS

JS.apply_style()
R = Path("/Users/sunmy/Documents/QF/GWAS/output/core_results")
O = Path("/Users/sunmy/Documents/QF/GWAS/output/figures_v3")

d = pd.read_csv(R / "faceo_r_enhanced.csv")
piv = d.pivot_table(index="condition", columns="algo", values="coh", aggfunc="mean")
ORDER = ["clean_0.0", "dropout_0.5", "noise_0.5", "noise_1.0", "samples_0.15", "samples_0.3"]
LBL = ["Clean", "Dropout 50%", "Noise 0.5", "Noise 1.0", "Samples 15%", "Samples 30%"]
piv = piv.reindex([c for c in ORDER if c in piv.index])

fig = plt.figure(figsize=(JS.DOUBLE, 2.6))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.0], wspace=0.30)
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(piv)); w = 0.27
SER = [("WGCNA-style", JS.C_WGCNA, "WGCNA-style"), ("Leiden γ=1.0", JS.C_LEIDEN, "Leiden"),
       ("FACEO-R (ours)", JS.C_OURS, "FACEO-R")]
for i, (k, c, lab) in enumerate(SER):
    if k in piv.columns:
        ax.bar(x + (i - 1) * w, piv[k].values, w, color=c, label=lab, edgecolor="white", linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels(LBL, rotation=32, ha="right")
ax.set_ylabel("Mean functional coherence")
ax.set_title("Across degradation regimes", fontsize=8.5)
ax.legend(frameon=False, fontsize=7, ncol=1, loc="upper right")
JS.tidy_axis(ax)
JS.panel_label(ax, "a")

ax2 = fig.add_subplot(gs[0, 1])
al = []
for col in d.algo.unique():
    if col.startswith("alpha="):
        al.append((float(col.split("=")[1]), d.loc[d.algo == col, "coh"].mean()))
al.sort()
ax2.plot([a for a, _ in al], [m for _, m in al], "o-", color=JS.C_OURS,
         ms=4, lw=1.3, label="Fixed $\\alpha$")
adapt = d.loc[d.algo == "FACEO-R (ours)", "coh"].mean()
ax2.axhline(adapt, ls="--", lw=1.2, color=JS.C_LEIDEN, label="Adaptive $\\alpha(q)$")
best = min(al, key=lambda t: t[1])
ax2.scatter([best[0]], [best[1]], s=45, facecolor="none", edgecolor="k", lw=1.2, zorder=5)
ax2.annotate("optimum", (best[0], best[1]), textcoords="offset points",
             xytext=(8, -10), fontsize=7)
ax2.set_xlabel("Anchoring weight $\\alpha$ (fixed)")
ax2.set_ylabel("Mean functional coherence")
ax2.set_title("Anchoring-strength sweep", fontsize=8.5)
ax2.legend(frameon=False, fontsize=7, loc="lower left")
JS.tidy_axis(ax2)
JS.panel_label(ax2, "b")
fig.savefig(O / "FigCA_clean.pdf")
fig.savefig(O / "FigCA_clean.png", dpi=600)
plt.close(fig)
print("✅ FigCA_clean (双栏)")
