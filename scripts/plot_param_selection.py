#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总「自动选参策略」的四类实验结果 -> 论文级表格 + SCI 图
（全部数据来自已完成的实验，无新计算）
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
BASE = Path("/Users/sunmy/Documents/QF/GWAS")
OUT = BASE / "output/core_results"
FIG = BASE / "output/figures_v3"

# 来自各实验的真实数值
ROWS = [
    {"strategy": "Structure fingerprint (RandomForest)",
     "type": "Regression on structural features", "n": 70,
     "metric": "accuracy", "value": 0.171, "baseline": 0.167,
     "gain_vs_fixed_gamma": -0.149, "verdict": "failed"},
    {"strategy": "Structure fingerprint (LogisticReg)",
     "type": "Regression on structural features", "n": 70,
     "metric": "accuracy", "value": 0.157, "baseline": 0.167,
     "gain_vs_fixed_gamma": -0.167, "verdict": "failed"},
    {"strategy": "Pathway-size binning rule",
     "type": "Single-feature grouping rule", "n": 70,
     "metric": "spearman_rho", "value": 0.222, "baseline": 0.0,
     "gain_vs_fixed_gamma": np.nan, "verdict": "failed (p=0.064)"},
    {"strategy": "Multi-resolution consensus clustering",
     "type": "Aggregation over resolutions", "n": 12,
     "metric": "win_rate", "value": 0.0, "baseline": 0.5,
     "gain_vs_fixed_gamma": -2.226, "verdict": "failed"},
    {"strategy": "Cross-library held-out (GO to Reactome)",
     "type": "Annotation-guided selection", "n": 11,
     "metric": "win_rate", "value": 0.455, "baseline": 0.5,
     "gain_vs_fixed_gamma": 0.286, "verdict": "unstable"},
]

# 各组合的最优 gamma 分布（用于图）
GAMMA_DIST = {0.1: 6, 0.5: 18, 1.0: 13, 2.0: 13, 5.0: 10, 10.0: 10}


def main():
    d = pd.DataFrame(ROWS)
    d.to_csv(OUT / "parameter_selection_attempts.csv", index=False)
    print(f"已保存 parameter_selection_attempts.csv ({len(d)} 条策略)")

    bt = pd.DataFrame([{"gamma": k, "count": v,
                        "fraction": v / sum(GAMMA_DIST.values())}
                       for k, v in GAMMA_DIST.items()])
    bt.to_csv(OUT / "optimal_gamma_distribution.csv", index=False)
    from scipy.stats import entropy
    ent = entropy(bt["fraction"].values, base=np.e)
    print(f"最优 γ 分布熵 = {ent:.3f} (最大 {np.log(6):.3f}) → "
          f"{'高度分散，无普适最优' if ent > 1.6 else '集中'}")
    print(bt.to_string(index=False))

    # ---------------- 图：两面板 ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MM = 1 / 25.4
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.6,
        "axes.linewidth": 0.6, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 2.3))
    fig.subplots_adjust(wspace=0.45)

    # (a) 最优 γ 分布
    ax = axes[0]
    gs = list(GAMMA_DIST.keys())
    cnt = [GAMMA_DIST[g] for g in gs]
    cols = ["#666666"] * len(gs)
    cols[gs.index(1.0)] = "#D55E00"
    ax.bar(range(len(gs)), cnt, color=cols, width=0.68)
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([f"{g:g}" for g in gs])
    ax.set_xlabel("Resolution $\\gamma$")
    ax.set_ylabel("Occurrences (best per pathway)")
    ax.set_title(f"No universal optimum\n(distribution entropy {ent:.2f} / max {np.log(6):.2f})",
                 fontsize=7.2)
    for i, c in enumerate(cnt):
        ax.text(i, c + 0.3, str(c), ha="center", fontsize=6)
    ax.axhline(np.mean(cnt), ls=":", color="#999999", lw=0.7)
    ax.text(-0.28, 1.03, "a", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")

    # (b) 四类策略的失败
    ax = axes[1]
    labels = ["Struct.\nfingerprint\n(RF)", "Struct.\nfingerprint\n(LR)",
              "Size\nbinning", "Multi-res.\nconsensus", "Held-out\n(GO→React.)"]
    gains = [-0.149, -0.167, np.nan, -2.226, 0.286]
    xs = np.arange(len(labels))
    vals = [g if g == g else 0 for g in gains]
    colors = ["#D55E00" if (g == g and g < 0) else ("#0072B2" if g == g else "#CCCCCC")
              for g in gains]
    ax.bar(xs, vals, color=colors, width=0.62)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=5.3)
    ax.set_ylabel("Gain vs fixed $\\gamma$=1.0")
    ax.set_title("All selection strategies fail\n(consensus collapses)", fontsize=7.2)
    ax.annotate("no significant\ncorrelation (p=0.064)", xy=(2, 0), xytext=(1.5, -1.5),
                fontsize=5.0, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.5, color="#666666"))
    ax.text(-0.30, 1.03, "b", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")

    o = FIG / "Fig10_parameter_selection.pdf"
    fig.savefig(o); fig.savefig(o.with_suffix(".png"), dpi=600); plt.close(fig)
    print(f"✅ {o.name}")


if __name__ == "__main__":
    main()
