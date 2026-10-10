#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FACEO vs Leiden vs WGCNA — held-out 公平对比实验

公平性：
  建图用 GO BP（FACEO）/ 仅表达（Leiden, WGCNA）
  评价用 held-out Reactome（三者都没在建模时使用）

评价指标（对硬划分与重叠划分定义一致）：
  gene_coverage = 每个基因的 Reactome 注释被其所属模块注释并集覆盖的比例
  module_purity = 模块内最高频 Reactome 注释的占比
  score = coverage × purity
"""
import re
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import gwas_full_analysis as GA
import gwas_full_analysis_v3 as V3
import gwas_algorithms_v2 as A2
import gwas_faceo as FC

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
GS = Path("/Users/sunmy/.hermes/gwas_data")
OUT = BASE / "output/core_results"


def to_g2t(terms):
    d = defaultdict(set)
    for t, gs in terms.items():
        for g in gs:
            d[str(g).upper()].add(t)
    return d


def eval_modules(modules, genes, ref2t):
    """统一的评价函数（硬划分传单归属模块，重叠划分传多归属模块）"""
    if not modules:
        return dict(coverage=np.nan, purity=np.nan, score=np.nan,
                    n_modules=0, mean_membership=0.0)
    # 基因 -> 所属模块集合
    g2mods = defaultdict(list)
    for mi, m in enumerate(modules):
        for i in m:
            g2mods[i].append(mi)
    # 覆盖率（基因层面）
    covs = []
    for i, g in enumerate(genes):
        tg = ref2t.get(g.upper(), set())
        if not tg:
            continue
        ms = g2mods.get(i, [])
        if not ms:
            covs.append(0.0); continue
        union = set()
        for mi in ms:
            for j in modules[mi]:
                union |= ref2t.get(genes[j].upper(), set())
        covs.append(len(tg & union) / len(tg))
    coverage = float(np.mean(covs)) if covs else np.nan
    # 纯度
    purs = []
    for m in modules:
        cnt = Counter()
        for i in m:
            for t in ref2t.get(genes[i].upper(), set()):
                cnt[t] += 1
        if cnt:
            purs.append(max(cnt.values()) / len(m))
    purity = float(np.mean(purs)) if purs else np.nan
    mean_memb = float(np.mean([len(v) for v in g2mods.values()])) if g2mods else 0.0
    score = coverage * purity if (coverage == coverage and purity == purity) else np.nan
    return dict(coverage=coverage, purity=purity, score=score,
                n_modules=len(modules), mean_membership=mean_memb)


def main(max_cohorts=5):
    go_bp = json.load(open(GS / "GO_Biological_Process_2021.json"))
    reactome = json.load(open(GS / "Reactome_2022.json"))
    go2t, ref2t = to_g2t(go_bp), to_g2t(reactome)
    print(f"GO-BP {len(go_bp)} terms | Reactome {len(reactome)} terms（held-out）")

    res = pd.read_csv(OUT / "full_eval_v3.csv")
    res["algo"] = res["algo"].astype(str)
    cohorts = sorted(res.gse.unique())[:max_cohorts]
    fn = dict(GA.ALGOS); fn.update(A2.NEW_ALGOS)
    rows = []
    t0 = time.time()
    for gse in cohorts:
        try:
            raw = pd.read_csv(BASE / f"qcdata/{gse}.csv", index_col=0,
                              quoting=3, low_memory=False)
            raw.index = [str(i).strip().strip('"').upper() for i in raw.index]
            keep = [re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(i)) is not None
                    for i in raw.index]
            df = V3.preprocess_robust(raw[keep])
        except Exception as e:
            print(f"  skip {gse}: {e}"); continue
        go_all = V3.load_go_sets()
        paths = V3.select_pathways(go_all, set(df.index))
        for pw_name, pw_genes in paths[:5]:
            genes = [g for g in pw_genes if g in df.index]
            if len(genes) < 60 or len(genes) > 250:
                continue
            genes = list(genes)
            tom = GA.compute_tom(df.loc[genes])
            g = GA.to_ig(tom, genes)

            # ---- 三种方法 ----
            lab_w = np.asarray(fn["WGCNA-style\nHierarchical"](g))
            lab_l = np.asarray(fn["Leiden γ=1.0"](g))
            modules_f, hard_f, memb_f = FC.faceo(tom, genes, go2t,
                                                 alpha=0.6, gamma=1.0, tau=0.65)

            for name, mods, extra in [
                ("WGCNA-style", FC.partition_to_modules(lab_w), {}),
                ("Leiden γ=1.0", FC.partition_to_modules(lab_l), {}),
                ("FACEO (ours)", modules_f, {"mean_memb": None}),
            ]:
                ev = eval_modules(mods, genes, ref2t)
                rows.append({"gse": gse, "pathway": pw_name[:28], "algo": name,
                             "n_genes": len(genes), **ev})
            if len(rows) % 15 == 0:
                print(f"  ... {len(rows)} 条 ({time.time()-t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "faceo_comparison.csv", index=False)
    print(f"\n共 {len(d)} 条记录 ({d.groupby(['gse','pathway']).ngroups} 组合)")
    agg = d.groupby("algo").agg(
        coverage=("coverage", "mean"), purity=("purity", "mean"),
        score=("score", "mean"), n_modules=("n_modules", "mean"),
        mean_memb=("mean_membership", "mean")).round(4)
    print("\n=== 三方法对比（held-out Reactome 评价）===")
    print(agg.to_string())
    agg.to_csv(OUT / "faceo_summary.csv")

    # 逐组合胜率
    piv = d.pivot_table(index=["gse", "pathway"], columns="algo", values="score")
    for a in ["Leiden γ=1.0", "WGCNA-style"]:
        if a in piv.columns and "FACEO (ours)" in piv.columns:
            w = (piv["FACEO (ours)"] > piv[a]).sum()
            print(f"FACEO 优于 {a}: {w}/{len(piv)} 组合")
    return d


if __name__ == "__main__":
    main()
