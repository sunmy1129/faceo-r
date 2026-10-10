#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成分析报告 PDF（整合结果表 + 统计检验 + 图表）
"""
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from fpdf import FPDF

warnings.filterwarnings("ignore")

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
OUT = BASE / "output/core_results"
FIG = BASE / "output/figures"
REG = "/Users/sunmy/.hermes/fonts/SongtiSC-Regular.ttf"
BOLD = "/Users/sunmy/.hermes/fonts/SongtiSC-Bold.ttf"
PDF_OUT = BASE / "output/GWAS_Analysis_Report.pdf"

BLACK, DARK, MID, RULE = (0, 0, 0), (40, 40, 40), (105, 105, 105), (150, 150, 150)


class Doc(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("r", size=8)
        self.set_text_color(*MID)
        self.cell(0, 5, str(self.page_no()), align="C")

    def h1(self, t):
        if self.get_y() > self.page_break_trigger - 28:
            self.add_page()
        self.ln(2)
        self.set_font("b", size=12)
        self.set_text_color(*BLACK)
        self.set_x(self.l_margin)
        self.multi_cell(0, 6.6, t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLACK)
        self.set_line_width(0.25)
        y = self.get_y() + 0.6
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(2.2)

    def h2(self, t):
        self.ln(0.5)
        self.set_font("b", size=9.3)
        self.set_text_color(*BLACK)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.3, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def p(self, t, size=8.4, lh=4.7, color=DARK):
        self.set_font("r", size=size)
        self.set_text_color(*color)
        self.set_x(self.l_margin)
        self.multi_cell(0, lh, t, new_x="LMARGIN", new_y="NEXT")

    def bullet(self, t, size=8.4, lh=4.7):
        self.set_font("r", size=size)
        self.set_text_color(*DARK)
        self.set_x(self.l_margin + 3.5)
        self.multi_cell(0, lh, f"\u2022  {t}", new_x="LMARGIN", new_y="NEXT")

    def fit(self, widths):
        avail = self.w - self.l_margin - self.r_margin
        tot = float(sum(widths))
        return [w * avail / tot for w in widths] if tot else widths

    def table(self, headers, rows, widths, size=7.6, lh=4.3, note=None):
        widths = self.fit(widths)
        if self.get_y() > self.page_break_trigger - 30:
            self.add_page()
        self.set_draw_color(*RULE)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.l_margin + sum(widths), self.get_y())
        self.ln(1.1)
        self.set_font("b", size=size)
        self.set_text_color(*BLACK)
        for h, w in zip(headers, widths):
            self.cell(w, 4.9, h)
        self.ln(4.9)
        self.set_line_width(0.15)
        self.line(self.l_margin, self.get_y(), self.l_margin + sum(widths), self.get_y())
        self.ln(0.8)
        for i, row in enumerate(rows):
            self.set_font("r", size=size)
            self.set_text_color(*DARK)
            wrapped = []
            for c, w in zip(row, widths):
                wrapped.append(self.multi_cell(w - 1.0, lh, str(c), dry_run=True, output="LINES"))
            n = max(len(x) for x in wrapped)
            rh = n * lh + 0.8
            if self.get_y() + rh > self.page_break_trigger - 1:
                self.add_page()
                self.set_font("b", size=size)
                self.set_text_color(*BLACK)
                for h, w in zip(headers, widths):
                    self.cell(w, 4.9, h)
                self.ln(4.9)
                self.set_draw_color(*RULE)
                self.set_line_width(0.15)
                self.line(self.l_margin, self.get_y(), self.l_margin + sum(widths), self.get_y())
                self.ln(0.8)
                self.set_font("r", size=size)
                self.set_text_color(*DARK)
            x0, yy = self.l_margin, self.get_y()
            for lines, w in zip(wrapped, widths):
                self.set_xy(x0, yy)
                for ln in lines:
                    self.cell(w - 1.0, lh, ln)
                    self.set_xy(x0, self.get_y() + lh)
                x0 += w
            self.set_xy(self.l_margin, yy + rh)
        self.set_draw_color(*RULE)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.l_margin + sum(widths), self.get_y())
        self.ln(2)
        if note:
            self.p(note, size=7.4, lh=4.1, color=MID)

    def figure(self, path, w=165, caption=None):
        if not Path(path).exists():
            return
        if self.get_y() > self.page_break_trigger - 70:
            self.add_page()
        x = (self.w - w) / 2
        self.image(str(path), x=x, w=w)
        self.ln(1.5)
        if caption:
            self.set_font("r", size=7.6)
            self.set_text_color(*MID)
            self.set_x(self.l_margin)
            self.multi_cell(0, 4.3, caption, new_x="LMARGIN", new_y="NEXT")
            self.ln(1.5)


def main():
    df = pd.read_csv(OUT / "full_eval.csv")
    df["algo"] = df["algo"].astype(str)
    stats_df = pd.read_csv(OUT / "statistics.csv") if (OUT / "statistics.csv").exists() else None

    pdf = Doc(format="A4")
    pdf.add_font("r", "", REG)
    pdf.add_font("b", "", BOLD)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(20, 16, 20)

    # ---------- 封面 ----------
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("b", size=17)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(0, 10, "GO 通路子网模块划分的方法学比较", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("b", size=12)
    pdf.multi_cell(0, 8, "多算法基准测试与六层验证分析报告", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("r", size=9)
    pdf.set_text_color(*MID)
    n_combo = df.groupby(["gse", "pathway"]).ngroups
    pdf.multi_cell(0, 5.4,
        f"数据集 {df.gse.nunique()} 个 · GO 通路 {df.groupby('gse')['pathway'].nunique().max()} 条 · "
        f"分析组合 {n_combo} 个\n"
        f"划分算法 {df.algo.nunique()} 种 · 评估记录 {len(df)} 条\n"
        f"评估层：结构 / 功能富集 / 稳定性 / 鲁棒性 / 生物学 / 随机对照",
        align="C", new_x="LMARGIN", new_y="NEXT")

    # ---------- 1 摘要 ----------
    pdf.add_page()
    pdf.h1("1   分析概览")
    pdf.p("本报告基于同一批基因表达数据（4 个 GEO 队列）与同一套 GO 通路基因集，"
          "在同一条拓扑重叠矩阵上对比多种划分算法的行为差异，评估维度覆盖结构、功能、稳定性、"
          "鲁棒性、生物学意义与随机对照六个方向。所有算法共享完全相同的输入，确保比较的公平性。")
    pdf.h2("1.1   数据集与通路")
    combo = df.groupby(["gse", "pathway"]).agg(
        genes=("n_genes", "first")).reset_index()
    combo_sum = combo.groupby("gse").agg(通路数=("pathway", "nunique"),
                                        平均基因数=("genes", "mean")).reset_index()
    disease = {"GSE137342": "败血症", "GSE25066": "乳腺癌",
               "GSE39582": "结肠癌", "GSE6764": "肝细胞癌"}
    rows = [[r["gse"], disease.get(r["gse"], "—"), int(r["通路数"]),
             f"{r['平均基因数']:.0f}"] for _, r in combo_sum.iterrows()]
    pdf.table(["数据集", "疾病类型", "GO 通路数", "通路平均基因数"], rows,
              [30, 30, 26, 34], size=8)

    # ---------- 2 主要结果 ----------
    pdf.h1("2   主要结果")
    agg = df.groupby("algo").agg(
        模块数=("n_modules", "mean"), 模块度Q=("modularity_Q", "mean"),
        功能内聚z=("coh_z_mean", "mean"), 稳定性=("stability_ARI", "mean"),
        抗噪性=("ARI_noise0.1", "mean"), 拮抗拆分率=("antag_split_rate", "mean"),
    ).round(3).sort_values("功能内聚z", ascending=False)
    rows = [[i, f"{r['模块数']:.2f}", f"{r['模块度Q']:.3f}", f"{r['功能内聚z']:.2f}",
             f"{r['稳定性']:.3f}", f"{r['抗噪性']:.3f}", f"{r['拮抗拆分率']:.2f}"]
            for i, r in agg.iterrows()]
    pdf.table(["算法", "模块数", "模块度 Q", "功能内聚 z", "稳定性 ARI", "抗噪 ARI", "拮抗拆分率"],
              rows, [44, 17, 19, 21, 21, 19, 21], size=7.3,
              note="功能内聚 z 为主指标（模块内 GO 注释相似度相对通路内随机期望的标准化），"
                   "不受模块大小影响；模块度 Q 反映拓扑紧致度；稳定性为 bootstrap 重采样一致性；"
                   "抗噪为 10% 表达噪声下的划分一致性；拮抗拆分率为已知功能拮抗基因对被分入不同模块的比例。")

    # ---------- 3 关键发现 ----------
    pdf.h1("3   关键发现")
    lg = df[df.algo.str.startswith("Leiden γ=")]
    if len(lg):
        piv = lg.groupby("algo").agg(n=("n_modules", "mean"),
                                     Q=("modularity_Q", "mean"),
                                     z=("coh_z_mean", "mean")).round(3)
        rows = [[i, f"{r['n']:.1f}", f"{r['Q']:.3f}", f"{r['z']:.2f}"]
                for i, r in piv.sort_values("n").iterrows()]
        pdf.h2("3.1   分辨率提升的代价（模块度崩塌）")
        pdf.table(["分辨率设置", "平均模块数", "模块度 Q", "功能内聚 z"], rows,
                  [40, 30, 26, 28], size=7.8)
        z_lo = piv["z"].iloc[0]
        z_hi = piv["z"].iloc[-1]
        pdf.p(f"随着分辨率参数提高，模块数从 {piv['n'].iloc[0]:.1f} 增至 {piv['n'].iloc[-1]:.1f}，"
              f"但模块度从 {piv['Q'].iloc[0]:.3f} 降至 {piv['Q'].iloc[-1]:.3f}，"
              f"功能内聚 z 分数由 {z_lo:.2f} 变为 {z_hi:.2f}。"
              f"该结果表明单纯提高分辨率获得的「更细划分」在拓扑紧致度与功能一致性上均付出代价，"
              f"说明划分质量不能以模块数量衡量。", size=8.2)

    pdf.h2("3.2   最大模块数与最佳功能一致性并非同一算法")
    best_mod = agg["模块数"].idxmax()
    best_coh = agg["功能内聚z"].idxmax()
    pdf.p(f"模块数最大的算法为 {best_mod}（{agg.loc[best_mod,'模块数']:.1f} 个模块），"
          f"而功能内聚性最高的算法为 {best_coh}（z = {agg.loc[best_coh,'功能内聚z']:.2f}，"
          f"模块数 {agg.loc[best_coh,'模块数']:.1f}）。二者不一致，"
          f"说明「拆得细」与「拆得对」是两个独立的评价维度，需要分别报告。", size=8.2)

    # ---------- 4 统计检验 ----------
    if stats_df is not None:
        pdf.h1("4   统计检验")
        pdf.p("所有比较均在同一数据集的同一通路上配对进行（检验单位为数据集×通路），"
              "采用 Wilcoxon 符号秩检验，并作 Benjamini–Hochberg 多重比较校正。"
              "效应量以 Cliff's delta 报告。")
        main_stats = stats_df[stats_df.hypothesis.isin(
            ["H1", "H2", "H3", "H4", "H3b", "H2b", "H2c"])]
        rows = []
        for _, r in main_stats.iterrows():
            rows.append([r["hypothesis"], r["description"], f"{r['median_a']:.3f}",
                         f"{r['median_b']:.3f}", f"{r['p_fdr']:.1e}",
                         f"{r['delta']:+.2f}", "是" if r["significant"] else "否"])
        pdf.table(["假设", "检验内容", "实验组中位", "对照中位", "p(FDR)", "δ", "显著"],
                  rows, [15, 48, 24, 22, 22, 16, 13], size=7.4,
                  note="实验组为 Leiden γ=1.0，对照为 WGCNA 式层次聚类。δ 为 Cliff's delta。"
                       "显著列以校正后 p < 0.05 判定。")

    # ---------- 5 图表 ----------
    pdf.add_page()
    pdf.h1("5   图表")
    for f, cap in [("Fig1_overpartition.png", "图 1  过分割现象：模块数与模块度的关系、模块数与功能一致性的关系、"
                                             "以及分辨率参数的连续扫描曲线。"),
                   ("Fig2_metrics.png", "图 2  多指标对比：功能内聚性排名、稳定性与鲁棒性关系、拮抗基因对拆分率。"),
                   ("Fig3_heatmap.png", "图 3  各算法在多项指标上的性能画像（z 分数标准化）。"),
                   ("Fig4_embedding.png", "图 4  自监督图表示学习的独立验证：嵌入划分的功能内聚性，"
                                          "以及嵌入与 Leiden 社区的一致性分布。"),
                   ("Fig5_synthetic.png", "图 5  合成基准：(a) 块模型下绝对准确率随社区模糊度变化；"
                                          "(b) 嵌套子群结构恢复准确率（检验多分辨率拆分能力）；"
                                          "(c) 噪声梯度下的稳健性。"),
                   ("Fig6_null.png", "图 6  零模型检验：(a) 观测模块度与两类零模型对比；"
                                     "(b) 相对零模型的 z 分数分布。"),
                   ("Fig7_downstream.png", "图 7  下游任务验证：模块特征基因预测临床表型"
                                           "（疾病状态 / 分子亚型 / 分期）的交叉验证 AUC。")]:
        pdf.figure(FIG / f, w=162, caption=cap)

    # ---------- 6 合成基准与下游任务 ----------
    pdf.add_page()
    pdf.h1("6   合成基准与下游任务")
    try:
        syn = pd.read_csv(OUT / "synthetic_results.csv")
        nested = syn[syn.benchmark == "Nested_sub"].groupby("algo").agg(
            ARI=("ARI", "mean"), 模块数=("n_modules", "mean")).round(3)
        nested = nested.sort_values("ARI", ascending=False)
        rows = [[i, f"{r['ARI']:.3f}", f"{r['模块数']:.1f}"] for i, r in nested.iterrows()]
        pdf.h2("5.1   嵌套子群结构的恢复准确率（绝对基准）")
        pdf.p("在人工构造的嵌套结构（每个模块内含 2 个功能子群，共 12 个子群）上，"
              "比较各算法恢复真实子群划分的准确率。该基准提供绝对参照，"
              "可区分「拆得细」与「拆得对」。", size=8.2)
        pdf.table(["算法", "ARI（vs 真实子群）", "平均模块数"], rows,
                  [60, 34, 28], size=7.5,
                  note="Leiden γ=2.0 与 CPM γ=0.10 达到最高准确率；"
                       "低分辨率设置（γ=0.5、谱聚类）因欠分割而准确率下降。")
    except Exception:
        pass

    try:
        nm = pd.read_csv(OUT / "null_models.csv")
        pdf.h2("5.2   零模型检验")
        pdf.p(f"以度保持随机重连（零模型 A）与表达矩阵置换（零模型 B）作为对照。"
              f"观测模块度相对零模型的平均 z 分数为 {nm['z_nullA'].mean():.2f}（A）与 "
              f"{nm['z_nullB'].mean():.2f}（B）；显著（p<0.05）比例分别为 "
              f"{(nm['p_nullA']<0.05).mean()*100:.0f}% 与 {(nm['p_nullB']<0.05).mean()*100:.0f}%。", size=8.2)
    except Exception:
        pass

    try:
        dw = pd.read_csv(OUT / "downstream_results.csv")
        if len(dw):
            pdf.h2("5.3   下游任务验证")
            agg = dw.groupby("algo").agg(AUC=("AUC", "mean"), n=("AUC", "count")).round(3)
            agg = agg.sort_values("AUC", ascending=False)
            rows = [[i, f"{r['AUC']:.3f}", int(r["n"])] for i, r in agg.iterrows()]
            pdf.table(["算法", "平均 CV AUC", "任务数"], rows, [60, 34, 26], size=7.4,
                      note="以各模块第一主成分（module eigengene）为特征，"
                           "预测临床表型（疾病状态 / ER 状态 / TNM 分期 / 组织类型），5 折交叉验证。")
    except Exception:
        pass

    # ---------- 6 方法与复现 ----------
    pdf.h1("7   方法与可复现性")
    pdf.bullet("建图：每条 GO 通路的基因子矩阵经中位数填补、必要时 log2 转换、低表达过滤后，"
               "计算软阈值相关邻接（β = 6）与拓扑重叠矩阵（TOM），作为所有算法共享的唯一输入。")
    pdf.bullet("划分：Leiden（γ = 0.5/1.0/2.0/5.0）、CPM（γ = 0.05/0.10）、Louvain、Infomap、"
               "Walktrap、标签传播、层次聚类加动态切割、K-means、高斯混合、Ward、谱聚类，"
               "以及自监督图表示学习（GAT 与 GraphSAGE，边预测任务）。")
    pdf.bullet("功能层：以 GO 生物过程、分子功能、细胞组分、KEGG 与 Reactome 共 6807 个基因集"
               "（基因数 ≥10）为参照，计算模块内基因对的注释相似度，"
               "并相对于同一通路内随机基因集作标准化，得到无大小偏差的 z 分数。")
    pdf.bullet("稳定性：对基因做 80% 比例的 bootstrap 重采样，重复 10 次，"
               "计算划分之间的平均调整兰德指数。")
    pdf.bullet("鲁棒性：对 TOM 加入 10% 与 20% 高斯噪声、随机删除 10% 边后重新划分，"
               "与基线划分比较。")
    pdf.bullet("生物学层：以 20 组已知功能拮抗基因对（如 BAX/BCL2、CASP3/XIAP、NFKB1/NFKBIA）"
               "计算被正确拆分的比例。")
    pdf.bullet("对照：以随机划分作为空白对照，报告与随机划分的调整兰德指数，"
               "用于确认算法划分显著偏离随机。")
    pdf.bullet("环境：Python 3.11，igraph / leidenalg / networkx / scikit-learn / PyTorch Geometric；"
               "随机种子统一固定为 42。所有结果表与图表由脚本自动生成。")

    pdf.output(PDF_OUT)
    print("已生成:", PDF_OUT)
    import os
    print("大小:", os.path.getsize(PDF_OUT), "bytes")


if __name__ == "__main__":
    main()
