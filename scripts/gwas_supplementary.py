#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Supplementary Material PDF（ISMB 投稿必需）"""
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
from gwas_report import Doc, REG, BOLD, BLACK, MID, DARK

warnings.filterwarnings("ignore")
BASE = Path("/Users/sunmy/Documents/QF/GWAS")
OUT = BASE / "output/core_results"
FIG = BASE / "output/figures"
FIG3 = BASE / "output/figures_v3"
PDF_OUT = BASE / "paper/supplementary.pdf"

DISEASE = {
    "GSE10072": "Lung adenocarcinoma", "GSE137342": "Severe sepsis",
    "GSE15653": "Type 2 diabetes / obesity", "GSE17536": "Colorectal cancer",
    "GSE19188": "Non-small cell lung cancer", "GSE25066": "Breast cancer",
    "GSE36961": "Hypertrophic cardiomyopathy", "GSE39582": "Colorectal cancer",
    "GSE42568": "Breast cancer", "GSE61635": "Systemic lupus erythematosus",
    "GSE6764": "Hepatocellular carcinoma",
}


def main():
    main_df = pd.read_csv(OUT / "full_eval_v3.csv")
    main_df["algo"] = main_df["algo"].astype(str)
    syn = pd.read_csv(OUT / "synthetic_results.csv")
    dw = pd.read_csv(OUT / "downstream_v3.csv")
    null = pd.read_csv(OUT / "null_models.csv")

    pdf = Doc(format="A4")
    pdf.add_font("r", "", REG)
    pdf.add_font("b", "", BOLD)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(18, 16, 18)

    # 封面
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("b", size=15)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(0, 9, "Supplementary Material", align="C",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("r", size=10)
    pdf.multi_cell(0, 6, "Systematic benchmarking of functional module partitioning\n"
                         "in GO pathway subnetworks", align="C",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("r", size=8.5)
    pdf.set_text_color(*MID)
    pdf.multi_cell(0, 5,
                   f"Supplementary Tables S1--S5\n"
                   f"Supplementary Figures S1--S2\n"
                   f"Total evaluation records: {len(main_df)} (main) + {len(syn)} (synthetic) "
                   f"+ {len(dw)} (downstream)",
                   align="C", new_x="LMARGIN", new_y="NEXT")

    # 附表 S1：数据集详情
    pdf.add_page()
    pdf.h1("Table S1. Cohort characteristics and pathway coverage")
    rows = []
    for gse in sorted(main_df.gse.unique()):
        s = main_df[main_df.gse == gse]
        rows.append([gse, DISEASE.get(gse, "—"), int(s.n_genes.median()),
                     int(s.n_samples.iloc[0]), int(s.pathway.nunique()),
                     int(len(s) / s.algo.nunique())])
    pdf.table(["Accession", "Disease", "Median pathway genes", "Samples",
               "Pathways", "Algorithm evaluations"],
              rows, [22, 52, 30, 18, 18, 30], size=7.4)

    # 附表 S2：算法完整指标
    pdf.add_page()
    pdf.h1("Table S2. Complete algorithm performance metrics")
    agg = main_df.groupby("algo").agg(
        n_mod=("n_modules", "mean"), Q=("modularity_Q", "mean"),
        coh=("coh_z_mean", "mean"), stab=("stability_ARI", "mean"),
        rob=("ARI_noise0.1", "mean"), ant=("antag_split_rate", "mean"),
        n=("n_modules", "count")).round(3).sort_values("coh", ascending=False)
    rows = []
    for i, r in agg.iterrows():
        f = lambda v: f"{v:.3f}" if v == v else "—"
        rows.append([i, f(r["n_mod"]), f(r["Q"]), f(r["coh"]), f(r["stab"]),
                     f(r["rob"]), f(r["ant"]), int(r["n"])])
    pdf.table(["Algorithm", "Modules", "Modularity Q", "Coherence z",
               "Stability", "Robustness", "Antagonistic", "n"],
              rows, [48, 18, 20, 20, 18, 18, 20, 12], size=6.6)

    # 附表 S3：统计检验
    st = pd.read_csv(OUT / "statistics_v3.csv")
    pdf.add_page()
    pdf.h1("Table S3. Paired statistical tests")
    rows = [[r["description"], int(r["n_pairs"]), f"{r['median_experimental']:.3f}",
             f"{r['median_reference']:.3f}", f"{r['p_value']:.2e}",
             f"{r['p_fdr']:.2e}", f"{r['cliffs_delta']:+.3f}",
             "yes" if r["significant"] else "no"] for _, r in st.iterrows()]
    pdf.table(["Metric", "Pairs", "Median (exp)", "Median (ref)", "p",
               "q (FDR)", "Cliff δ", "Sig."],
              rows, [40, 14, 22, 22, 20, 20, 18, 12], size=7.0,
              note="Experimental: Leiden γ=1.0. Reference: WGCNA-style hierarchical. "
                   "Wilcoxon signed-rank, Benjamini-Hochberg correction.")

    # 附表 S4：合成基准
    pdf.add_page()
    pdf.h1("Table S4. Synthetic benchmark results (ground truth available)")
    lfr = syn[syn.benchmark == "LFR"].groupby("algo")["ARI"].agg(["mean", "max"]).round(3)
    nest = syn[syn.benchmark == "Nested_sub"].groupby("algo").agg(
        ARI_sub=("ARI", "mean"), n_mod=("n_modules", "mean")).round(3)
    j = lfr.join(nest, how="outer").sort_values("ARI_sub", ascending=False)
    rows = [[i, f"{r['mean']:.3f}" if r["mean"] == r["mean"] else "—",
             f"{r['max']:.3f}" if r["max"] == r["max"] else "—",
             f"{r['ARI_sub']:.3f}" if r["ARI_sub"] == r["ARI_sub"] else "—",
             f"{r['n_mod']:.1f}" if r["n_mod"] == r["n_mod"] else "—"]
            for i, r in j.iterrows()]
    pdf.table(["Algorithm", "LFR mean ARI", "LFR max ARI",
               "Nested sub-group ARI", "Modules"],
              rows, [48, 26, 24, 34, 20], size=6.8,
              note="LFR: block model across fuzziness μ ∈ {0.10–0.70}. "
                   "Nested: recovery of 12 ground-truth sub-groups (the decisive benchmark).")

    # 附表 S5：零模型
    pdf.add_page()
    pdf.h1("Table S5. Null-model comparison")
    rows = [[r["gse"], r["pathway"][:40], f"{r['Q_real']:.3f}",
             f"{r['Q_nullA_mean']:.3f}", f"{r['Q_nullB_mean']:.3f}",
             f"{r['z_nullA']:.2f}", f"{r['z_nullB']:.2f}"]
            for _, r in null.iterrows()]
    pdf.table(["Cohort", "Pathway", "Observed Q", "Null A (rewired)",
               "Null B (permuted)", "z (A)", "z (B)"],
              rows, [24, 58, 22, 26, 26, 16, 16], size=6.6,
              note=f"Mean z = {null['z_nullA'].mean():.2f} (A) and "
                   f"{null['z_nullB'].mean():.2f} (B). Negative z indicates that observed "
                   f"modularity does not exceed the null models.")

    # 附图
    pdf.add_page()
    pdf.h1("Supplementary Figures")
    for f, cap in [("Fig5_synthetic.png",
                    "Figure S1. Synthetic benchmark accuracy across fuzziness and noise levels."),
                   ("Fig6_null.png",
                    "Figure S2. Observed modularity versus null models.")]:
        pdf.figure(FIG / f, w=158, caption=cap)

    pdf.output(PDF_OUT)
    print(f"已生成: {PDF_OUT}")
    print(f"页数: {pdf.page_no()}  大小: {PDF_OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
