#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主分析引擎 v3：
- 算法：37 种（17 基础 + 20 扩展）
- 数据集：自动发现 qcdata 下全部队列
- 评估：六层（结构/功能/稳定性/鲁棒性/生物学/随机对照）
- 输出：full_eval_v3.csv（长表）
用法：python gwas_full_analysis_v3.py [数据集,逗号分隔|all] [bootstrap次数]
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import gwas_full_analysis as GA
import gwas_algorithms_v2 as A2
import gwas_l2_functional as L2

warnings.filterwarnings("ignore")

BASE = Path("/Users/sunmy/Documents/QF/GWAS")
DATA = BASE / "qcdata"
OUT = BASE / "output/core_results"
OUT.mkdir(parents=True, exist_ok=True)
GOSETS = Path("/Users/sunmy/.hermes/gwas_data")

# 7 条核心 GO 通路（沿用原设计）
PATHWAY_KEYS = [
    "apoptotic process", "inflammatory response", "regulation of immune response",
    "response to cytokine", "I-kappaB kinase/NF-kappaB", "oxidative stress",
    "chemokine-mediated signaling",
]

SEED = 42
np.random.seed(SEED)


def preprocess_robust(df):
    """自适应预处理：过滤阈值随数据分布调整，避免整表被滤空"""
    df = df.copy()
    df = df.fillna(df.median(axis=1), axis=0)
    if df.max().max() > 100:
        df = np.log2(df + 1)
    n_samp = df.shape[1]
    # 逐级放宽过滤条件
    for frac, q in [(0.5, 0.25), (0.3, 0.15), (0.2, 0.10), (0.1, 0.05)]:
        thr = float(df.quantile(q, axis=1).median())
        keep = (df > thr).sum(axis=1) > (n_samp * frac)
        if keep.sum() >= 1000:
            return df.loc[keep]
    return df


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_go_sets():
    import json as js
    terms = {}
    for lib in ("GO_Biological_Process_2021",):
        f = GOSETS / f"{lib}.json"
        if f.exists():
            d = js.load(open(f))
            for k, v in d.items():
                terms[k] = [g.upper() for g in v]
    return terms


def select_pathways(go_sets, expr_genes):
    """按关键词选出通路，取与表达数据重叠最大的那个"""
    picked = []
    for kw in PATHWAY_KEYS:
        cands = [(k, v) for k, v in go_sets.items() if kw.lower() in k.lower()]
        if not cands:
            continue
        scored = sorted(cands, key=lambda kv: len(set(kv[1]) & set(expr_genes)), reverse=True)
        name, genes = scored[0]
        ov = len(set(genes) & set(expr_genes))
        if ov >= 25:
            picked.append((name, [g for g in genes if g in expr_genes]))
    return picked


def discover_datasets():
    """自动发现 qcdata 下可用的表达矩阵（基因×样本，行数>5000）"""
    found = {}
    for f in sorted(DATA.glob("*.csv")):
        if f.name.endswith("_meta.csv"):
            continue
        try:
            head = pd.read_csv(f, index_col=0, nrows=5)
            ncol = head.shape[1]
            # 快速估算行数
            with open(f) as fh:
                nrow = sum(1 for _ in fh) - 1
            if nrow >= 1000 and ncol >= 15:
                found[f.stem] = {"rows": nrow, "cols": ncol}
        except Exception:
            continue
    return found


ALL_ALGOS = {}
ALL_ALGOS.update(GA.ALGOS)          # 17 种基础
ALL_ALGOS.update(A2.NEW_ALGOS)      # 20 种扩展
# 彻底移除高复杂度算法（igraph C 层在高节点数下不稳定/极慢）
EXCLUDE = {"Link Communities", "Edge Betweenness", "Optimal Modularity",
           "Markov Stability"}
for _a in list(ALL_ALGOS.keys()):
    if _a in EXCLUDE:
        del ALL_ALGOS[_a]

import signal as _signal  # 保留导入但不再用于中断 igraph 调用


ALGO_TIME_LIMIT = 12   # 秒
MAX_PATHWAY_GENES = 300
# 高复杂度算法：在大图上跳过（避免 igraph 长时间占用）
SLOW_ALGOS = {"Link Communities", "Edge Betweenness", "Optimal Modularity",
              "Markov Stability"}
SLOW_ALGO_MAX_N = 150   # 慢算法仅在此规模以下的通路上评估


def main(datasets=None, n_boot=10, with_embed=False):
    terms, allg, gene2terms = L2.load_genesets()
    universe = len(allg)
    go_sets = load_go_sets()
    log(f"基因集: {len(terms)} 功能项 | GO BP: {len(go_sets)} 集合 | 算法: {len(ALL_ALGOS)} 种")

    available = discover_datasets()
    if datasets and datasets != ["all"]:
        available = {k: v for k, v in available.items() if k in datasets}
    log(f"数据集: {len(available)} 个 -> {', '.join(available.keys())}")

    records = []
    t0 = time.time()
    for di, (gse, info) in enumerate(available.items(), 1):
        try:
            raw = pd.read_csv(DATA / f"{gse}.csv", index_col=0)
        except Exception as e:
            log(f"  ✗ {gse} 读取失败: {str(e)[:50]}")
            continue
        raw.index = [str(i).strip().strip('"').strip().upper() for i in raw.index]
        raw = raw[~raw.index.duplicated()]
        # 排除非基因符号行名（纯数字探针）
        import re as _re
        keep_mask = [_re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(i)) is not None
                     for i in raw.index]
        if sum(keep_mask) < 1000:
            log(f"  ⏭ {gse} 行名为探针 ID（非基因符号），跳过")
            continue
        raw = raw[keep_mask]
        df = preprocess_robust(raw)
        expr_genes = set(df.index)
        paths = select_pathways(go_sets, expr_genes)
        log(f"({di}/{len(available)}) {gse}: {df.shape[0]} 基因 × {df.shape[1]} 样本 | 命中通路 {len(paths)}")
        if not paths:
            continue

        for pw_name, pw_genes in paths:
            avail = [g for g in pw_genes if g in df.index]
            if len(avail) < 25:
                continue
            # 限制通路规模（按方差取 top N，控制计算量）
            if len(avail) > MAX_PATHWAY_GENES:
                v = df.loc[avail].var(axis=1).sort_values(ascending=False)
                avail = list(v.index[:MAX_PATHWAY_GENES])
            df_sub = df.loc[avail]
            genes = list(df_sub.index)
            tom = GA.compute_tom(df_sub)
            g = GA.to_ig(tom, genes)
            n_to = 0
            for aname, afn in ALL_ALGOS.items():
                # 大图上跳过高复杂度算法
                if aname in SLOW_ALGOS and len(genes) > SLOW_ALGO_MAX_N:
                    n_to += 1
                    continue
                try:
                    labels = np.asarray(afn(g))
                except Exception:
                    continue
                rec = {"gse": gse, "pathway": pw_name, "algo": aname.replace("\n", " "),
                       "n_genes": len(genes), "n_samples": df.shape[1]}
                rec.update(GA.eval_L1(g, labels))
                rec.update(L2.evaluate_functional(labels, genes, terms, universe,
                                                  gene2terms, n_perm=30))
                rec.update(GA.eval_antag(labels, genes))
                rec.update(GA.eval_random_ctrl(labels, genes, n_perm=30))
                rec["stability_ARI"] = (GA.eval_stability(tom, genes, afn, n_boot=n_boot)
                                        if aname not in SLOW_ALGOS else np.nan)
                if aname not in SLOW_ALGOS:
                    rec.update(GA.eval_robust(tom, genes, afn))
                records.append(rec)
            log(f"    {pw_name[:40]}: 累计 {len(records)} 条 ({time.time()-t0:.0f}s)")

    df_res = pd.DataFrame(records)
    # 按数据集输出独立文件（便于分批合并，避免单文件冲突）
    tag = "_".join(sorted(available.keys()))[:60] if len(available) <= 3 else "all"
    out_csv = OUT / f"v3_{tag}.csv"
    df_res.to_csv(out_csv, index=False)
    log(f"✅ 已保存 {out_csv.name} ({len(df_res)} 行, {time.time()-t0:.0f}s)")

    pd.set_option("display.width", 240)
    print()
    g = df_res.groupby("algo").agg(
        模块数=("n_modules", "mean"), 模块度Q=("modularity_Q", "mean"),
        功能内聚=("coh_z_mean", "mean"), 稳定性=("stability_ARI", "mean"),
        拮抗拆分=("antag_split_rate", "mean"), n=("n_modules", "count")).round(3)
    print(g.sort_values("功能内聚", ascending=False))
    return df_res


if __name__ == "__main__":
    import sys
    ds = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else None
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(datasets=ds, n_boot=nb)
