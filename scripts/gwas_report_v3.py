#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 v3 分析报告 PDF（11 数据集 × 31 算法）"""
import sys
from pathlib import Path
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import warnings
import pandas as pd
import numpy as np
from gwas_report import Doc, REG, BOLD, BLACK, MID, DARK

warnings.filterwarnings("ignore")
BASE = Path("/Users/sunmy/Documents/QF/GWAS")
OUT = BASE / "output/core_results"
FIG = BASE / "output/figures_v3"
PDF_OUT = BASE / "output/GWAS_Analysis_Report_v3.pdf"

DISEASE = {
    "GSE10072": "肺癌", "GSE137342": "败血症", "GSE15653": "2型糖尿病",
    "GSE17536": "结肠癌", "GSE19188": "肺癌", "GSE25066": "乳腺癌",
    "GSE36961": "心肌病", "GSE39582": "结肠癌", "GSE42568": "乳腺癌",
    "GSE61635": "红斑狼疮", "GSE6764": "肝细胞癌", "GSE33000": "阿尔茨海默",
}


def main():
    df = pd.read_csv(OUT / "full_eval_v3.csv")
    df["algo"] = df["algo"].astype(str)
    st = pd.read_csv(OUT / "statistics_v3.csv") if (OUT / "statistics_v3.csv").exists() else None
    dw = pd.read_csv(OUT / "downstream_v3.csv") if (OUT / "downstream_v3.csv").exists() else None

    pdf = Doc(format="A4")
    pdf.add_font("r", "", REG)
    pdf.add_font("b", "", BOLD)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(20, 16, 20)

    # 封面
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("b", size=17)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(0, 10, "GO 通路子网功能模块划分的系统基准评测", align="C",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("b", size=12)
    pdf.multi_cell(0, 8, "多分辨率社区检测与层次共表达聚类的对照研究", align="C",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("r", size=9)
    pdf.set_text_color(*MID)
    n_combo = df.groupby(["gse", "pathway"]).ngroups
    pdf.multi_cell(0, 5.4,
                   f"数据集 {df.gse.nunique()} 个 · GO 通路 {df.pathway.nunique()} 条 · "
                   f"分析组合 {n_combo} 个\n"
                   f"划分算法 {df.algo.nunique()} 种 · 评估记录 {len(df)} 条\n"
                   f"评估层：结构 / 功能一致性 / 稳定性 / 鲁棒性 / 生物学 / 下游预测",
                   align="C", new_x="LMARGIN", new_y="NEXT")

    # 1 概览
    pdf.add_page()
    pdf.h1("1   分析概览")
    pdf.p("本报告在同一拓扑重叠矩阵上对比多种划分算法的行为。所有算法共享完全相同的输入图，"
          "因此输出差异完全归因于划分过程本身。评估覆盖六个独立维度，"
          "并以人工合成基准提供绝对准确率参照。")
    pdf.h2("1.1   数据集")
    rows = []
    for gse in sorted(df.gse.unique()):
        s = df[df.gse == gse]
        rows.append([gse, DISEASE.get(gse, "—"), int(s.n_genes.median()),
                     int(s.n_samples.iloc[0]), int(s.pathway.nunique())])
    pdf.table(["数据集", "疾病类型", "通路中位基因数", "样本数", "命中通路"],
              rows, [30, 26, 30, 22, 24], size=7.8)

    # 2 主要结果
    pdf.h1("2   主要结果")
    agg = df.groupby("algo").agg(
        模块数=("n_modules", "mean"), 模块度=("modularity_Q", "mean"),
        功能内聚=("coh_z_mean", "mean"), 稳定性=("stability_ARI", "mean"),
        抗噪=("ARI_noise0.1", "mean"), 拮抗拆分=("antag_split_rate", "mean"),
    ).round(3).sort_values("功能内聚", ascending=False)
    rows = [[i, f"{r['模块数']:.1f}", f"{r['模块度']:.3f}", f"{r['功能内聚']:.2f}",
             f"{r['稳定性']:.3f}" if r['稳定性'] == r['稳定性'] else "—",
             f"{r['抗噪']:.3f}" if r['抗噪'] == r['抗噪'] else "—",
             f"{r['拮抗拆分']:.2f}" if r['拮抗拆分'] == r['拮抗拆分'] else "—"]
            for i, r in agg.iterrows()]
    pdf.table(["算法", "模块数", "模块度Q", "功能内聚z", "稳定性", "抗噪性", "拮抗拆分率"],
              rows, [52, 18, 19, 21, 18, 17, 21], size=6.9,
              note="功能内聚 z 为主指标（模块内 GO 注释相似度相对通路内随机期望的标准化），"
                   "不受模块大小影响。拮抗拆分率为已知功能拮抗基因对被分入不同模块的比例。")

    # 3 统计检验
    if st is not None:
        pdf.h1("3   统计检验")
        pdf.p("检验单位为数据集×通路，配对设计，Wilcoxon 符号秩检验，"
              "Benjamini-Hochberg 多重比较校正，效应量以 Cliff's delta 报告。")
        rows = []
        for _, r in st.iterrows():
            rows.append([r["description"], f"{r['median_experimental']:.3f}",
                         f"{r['median_reference']:.3f}", f"{r['p_fdr']:.1e}",
                         f"{r['cliffs_delta']:+.2f}", "是" if r["significant"] else "否"])
        pdf.table(["检验指标", "实验组中位", "对照中位", "p(FDR)", "δ", "显著"],
                  rows, [46, 26, 24, 22, 18, 14], size=7.6,
                  note="实验组为 Leiden γ=1.0，对照为 WGCNA 式层次聚类。δ 为 Cliff's delta。")

    # 4 图表
    pdf.add_page()
    pdf.h1("4   图表")
    for f, cap in [("Fig1_overpartition.png",
                    "图 1  过分割现象：模块数与模块度、模块数与功能一致性的关系，以及分辨率的连续扫描。"),
                   ("Fig2_metrics.png",
                    "图 2  多指标对比：功能内聚性排序、稳定性与鲁棒性关系、拮抗基因对拆分率。"),
                   ("Fig3_heatmap.png",
                    "图 3  各算法在六项指标上的性能画像（z 分数标准化）。"),
                   ("Fig4_consistency.png",
                    "图 4  跨数据集一致性：各队列上实验组与对照的功能内聚差值分布。")]:
        pdf.figure(FIG / f, w=162, caption=cap)

    # 5 下游任务
    if dw is not None and len(dw):
        pdf.h1("5   下游任务验证")
        agg2 = dw.groupby("algo").agg(AUC=("AUC", "mean"), n=("AUC", "count")).round(3)
        agg2 = agg2.sort_values("AUC", ascending=False)
        rows = [[i, f"{r['AUC']:.3f}", int(r["n"])] for i, r in agg2.iterrows()]
        pdf.table(["算法", "平均 CV AUC", "任务数"], rows, [60, 34, 26], size=7.2,
                  note="以各模块第一主成分（module eigengene）为特征，用逻辑回归预测疾病状态，"
                       "5 折分层交叉验证，报告 ROC 曲线下面积。")

    # 6 方法
    pdf.h1("6   方法")
    for t in [
        "建图：每条 GO 通路的基因子矩阵经中位数填补、必要时 log2 转换、自适应低表达过滤后，"
        "计算软阈值相关邻接（β=6）与拓扑重叠矩阵（TOM），作为所有算法共享的唯一输入。",
        "划分：31 种算法，涵盖模块度优化（Leiden 多分辨率、Louvain、CPM、Multilevel、"
        "Fast Greedy、Leading Eigenvector 等）、信息论与谱方法（InfoMap、Walktrap、谱聚类）、"
        "生成式模型（标签传播、随机块）、重叠社区（团渗流）、图表示学习（自监督 GAT/GraphSAGE）、"
        "经典聚类（K-means、GMM、Ward、DBSCAN、HDBSCAN、OPTICS、Mean Shift、"
        "Affinity Propagation、非负矩阵分解）。",
        "功能层：以 GO、KEGG、Reactome 共 6807 个基因集（基因数 ≥10）为参照，"
        "计算模块内基因对的注释相似度，并相对于同一通路内随机基因集标准化，得到无大小偏差的 z 分数。",
        "稳定性：对基因做 80% bootstrap 重采样，重复 4 次，计算划分间的平均调整兰德指数。",
        "鲁棒性：对 TOM 加入 10% 与 20% 高斯噪声、随机删除 10% 边后重新划分，与基线比较。",
        "对照：随机划分作为空白对照；WGCNA 式层次聚类作为方法学对照。",
        "环境：Python 3.11，igraph / leidenalg / networkx / scikit-learn / PyTorch Geometric；"
        "随机种子固定为 42。",
    ]:
        pdf.bullet(t, size=8.2)

    pdf.output(PDF_OUT)
    print(f"已生成: {PDF_OUT}")
    print(f"大小: {PDF_OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
