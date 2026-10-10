#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可行性检验：功能注释注入 + 重叠划分是否有机制性优势空间
检验 1: 基因多功能性（一个基因被多少 GO term 注释）
检验 2: GO / Reactome 注释覆盖率（能否支撑 held-out 验证）
检验 3: 硬划分被迫损失的功能信息量（上界估计）
"""
import json
import re
import sys
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import gwas_full_analysis_v3 as V3

GS = Path("/Users/sunmy/.hermes/gwas_data")
BASE = Path("/Users/sunmy/Documents/QF/GWAS")


def load(fn):
    f = GS / fn
    return json.load(open(f)) if f.exists() else {}


def to_g2t(terms):
    d = defaultdict(set)
    for t, gs in terms.items():
        for g in gs:
            d[str(g).upper()].add(t)
    return d


def main():
    go_bp = load("GO_Biological_Process_2021.json")
    go_mf = load("GO_Molecular_Function_2021.json")
    kegg = load("KEGG_2021_Human.json")
    reactome = load("Reactome_2022.json")
    go2t = to_g2t(go_bp); mf2t = to_g2t(go_mf)
    kg2t = to_g2t(kegg); re2t = to_g2t(reactome)
    print(f"基因集规模: GO-BP {len(go_bp)} | GO-MF {len(go_mf)} | KEGG {len(kegg)} | Reactome {len(reactome)}")

    # 检验 1 & 2：在真实通路上统计
    print("\n=== 检验 1 & 2：真实通路的注释统计 ===")
    cohorts = ["GSE10072", "GSE137342", "GSE15653", "GSE17536", "GSE19188"]
    rows = []
    for gse in cohorts:
        try:
            raw = pd.read_csv(BASE / f"qcdata/{gse}.csv", index_col=0, quoting=3,
                              low_memory=False)
            raw.index = [str(i).strip().strip('"').upper() for i in raw.index]
            keep = [re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(i)) is not None
                    for i in raw.index]
            df = V3.preprocess_robust(raw[keep])
        except Exception:
            continue
        go_all = V3.load_go_sets()
        paths = V3.select_pathways(go_all, set(df.index))
        for pw_name, pw_genes in paths[:5]:
            genes = [g for g in pw_genes if g in df.index]
            if len(genes) < 40:
                continue
            n_go = sum(1 for g in genes if g in go2t)
            n_re = sum(1 for g in genes if g in re2t)
            n_both = sum(1 for g in genes if g in go2t and g in re2t)
            multi = [len(go2t[g]) for g in genes if g in go2t]
            multi_re = [len(re2t[g]) for g in genes if g in re2t]
            rows.append({
                "gse": gse, "pathway": pw_name[:26], "n_genes": len(genes),
                "GO_cov": n_go / len(genes), "Reactome_cov": n_re / len(genes),
                "both_cov": n_both / len(genes),
                "mean_GO_terms": np.mean(multi) if multi else 0,
                "pct_multifunc_GO": np.mean([m > 1 for m in multi]) if multi else 0,
                "mean_Reactome": np.mean(multi_re) if multi_re else 0,
            })
    d = pd.DataFrame(rows)
    print(d.to_string(index=False))
    print(f"\n汇总（跨 {len(d)} 个通路-队列组合）:")
    print(f"  GO-BP 注释覆盖率      {d.GO_cov.mean():.1%}")
    print(f"  Reactome 注释覆盖率   {d.Reactome_cov.mean():.1%}")
    print(f"  两者都有注释          {d.both_cov.mean():.1%}")
    print(f"  平均每个基因的 GO term 数  {d.mean_GO_terms.mean():.1f}")
    print(f"  多功能基因比例（>1 term）  {d.pct_multifunc_GO.mean():.1%}")
    print(f"  平均每个基因的 Reactome 通路数  {d.mean_Reactome.mean():.1f}")
    d.to_csv(BASE / "output/core_results/annotation_coverage.csv", index=False)

    # 检验 3：硬划分 vs 重叠划分的信息量上界
    print("\n=== 检验 3：重叠划分的理论上界 ===")
    tot_multi = d.pct_multifunc_GO.mean()
    print(f"  若 {tot_multi:.0%} 的基因有多个功能归属，则硬划分平均丢弃")
    print(f"  {d.mean_GO_terms.mean() - 1:.1f} 个功能归属/基因（占其归属的 {(d.mean_GO_terms.mean()-1)/d.mean_GO_terms.mean():.0%}）")
    print(f"  → 重叠划分在'功能覆盖率'上有结构性优势空间")

    # held-out 独立性检查
    go_set = set(go2t); re_set = set(re2t)
    print(f"\n=== held-out 可行性 ===")
    print(f"  GO-BP 有注释的基因: {len(go_set)}")
    print(f"  Reactome 有注释的基因: {len(re_set)}")
    print(f"  交集: {len(go_set & re_set)}")
    print(f"  → {'✅ 可用 GO 建图 / Reactome 评价' if len(go_set & re_set) > 5000 else '⚠️ 覆盖不足'}")


if __name__ == "__main__":
    main()
