#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期刊级绘图规范 (Cell / Nature style)
—— 统一配色、排版、字体、线条、面板标注
参考: Nature Publishing Group (NPG) 配色 + Cell Press 排版习惯
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

MM = 1 / 25.4
SINGLE = 89 * MM      # 单栏
ONE_HALF = 120 * MM   # 1.5 栏
DOUBLE = 183 * MM     # 双栏

# ---------------------------------------------------------------- 配色板
# NPG (Nature Publishing Group) — 高区分度、印刷友好
NPG = ["#1D4E89", "#B5443A", "#2A7F62", "#C87B3F", "#7B5E8E",
       "#8A94A6", "#5B8C9E", "#A63A46", "#6B7C4E", "#A89B8C"]

# Cell Press 常用（柔和、低饱和）
CELL = ["#1D4E89", "#C87B3F", "#2A7F62", "#7B5E8E", "#5B8C9E",
        "#D8C9A3", "#A63A46", "#B0B8C0"]

# 方法专用配色（全文统一）
C_OURS = "#B5443A"     # 我们的方法 —— 深砖红
C_LEIDEN = "#1D4E89"   # Leiden —— 深靛蓝
C_WGCNA = "#8A94A6"    # WGCNA —— 蓝灰
C_CPM = "#2A7F62"      # CPM —— 深青绿
C_INFO = "#C87B3F"     # 信息论 —— 赭橙
C_CLASSIC = "#A89B8C"  # 经典聚类 —— 卡其灰

METHOD_COLORS = {
    "FACEO-R (ours)": C_OURS, "FACEO-R": C_OURS,
    "Leiden γ=1.0": C_LEIDEN, "Leiden": C_LEIDEN,
    "WGCNA-style": C_WGCNA, "WGCNA-style\nHierarchical": C_WGCNA,
}

# 顺序色阶（用于热图）
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seqblue", ["#FBFCFD", "#D3DEE8", "#8FAFC7", "#4A7FA5", "#1D4E89"])
DIV_RB = LinearSegmentedColormap.from_list(
    "divrb", ["#1D4E89", "#8A94A6", "#F5F5F3", "#C87B3F", "#B5443A"])


def apply_style():
    """应用全局样式（期刊级）"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue",
                            "DejaVu Sans"],
        "font.size": 5.8,
        "axes.labelsize": 6.2,
        "axes.titlesize": 7,
        "axes.linewidth": 0.45,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1a1a1a",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 1.8,
        "ytick.major.size": 1.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.fontsize": 5.2,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.25,
        "legend.borderpad": 0.1,
        "lines.linewidth": 0.65,
        "lines.markersize": 2.5,
        "patch.linewidth": 0.4,
        "grid.linewidth": 0.4,
        "grid.color": "#DDDDDD",
        "grid.alpha": 0.6,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "pdf.fonttype": 42,       # 可编辑矢量文字
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def panel_label(ax, letter, x=-0.16, y=1.06, size=8):
    """统一面板标注（粗体小写）"""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left", color="black")


def tidy_axis(ax, grid_axis=None):
    """统一坐标轴精修"""
    ax.tick_params(pad=1.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, lw=0.3, color="#E8E8E8", zorder=0)
        ax.set_axisbelow(True)


def sig_star(p):
    """显著性星号"""
    if p != p:
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def short_algo(name):
    """算法名缩写（图内用）"""
    n = str(name).replace("\n", " ")
    rep = {"WGCNA-style Hierarchical": "WGCNA", "WGCNA-style": "WGCNA",
           "FACEO-R (ours)": "FACEO-R", "FACEO (ours)": "FACEO",
           "SelfSup-GAT": "GAT", "SelfSup-SAGE": "SAGE",
           "Label propagation": "LabelProp", "Affinity Propagation": "AffProp",
           "Spectral clustering": "Spectral", "Gaussian mixture": "GMM",
           "Clique percolation": "CliquePerc", "Markov Stability": "MarkovStab",
           "Leading Eigenvector": "LeadEig", "Fast Greedy": "FastGreedy",
           "Edge Betweenness": "EdgeBetw", "Link Communities": "LinkComm",
           "Optimal Modularity": "OptMod"}
    for k, v in rep.items():
        if n.startswith(k):
            return v + n[len(k):]
    return n
