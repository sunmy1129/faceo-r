#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L6 下游任务扩展：自动识别临床标签，构建 5 类预测任务
1 疾病状态  2 分子亚型  3 分期  4 结局/预后  5 可成药性富集（Hub 基因）
输出：downstream_v3.csv
"""
import re
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import hypergeom

import gwas_full_analysis as GA
import gwas_algorithms_v2 as A2
import gwas_l2_functional as L2

warnings.filterwarnings("ignore")
BASE = Path("/Users/sunmy/Documents/QF/GWAS")
DATA = BASE / "qcdata"
OUT = BASE / "output/core_results"
SEED = 42

# 临床标签规则：名称 -> (正则, 阳性值, 阴性值)
RULES = [
    ("survival_outcome", r"(survival status|outcome|status)\s*:\s*([^|]+)", None, None),
    ("er_status", r"er[_ ]?status[^:]*:\s*([^|]+)", r"(pos|p\b|positive|\+)", r"(neg|n\b|negative|\-)"),
    ("stage", r"(tnm[\.\s]?stage|tumor stage|stage)\s*:\s*([^|]+)", r"(iii|iv|3|4)", r"(i\b|ii\b|1|2)"),
    ("tissue_type", r"(tissue|source|diagnosis)[^:]*:\s*([^|]+)", None, None),
]


def text_of(row):
    return (str(row.get("chars", "")) + " | " + str(row.get("source", "")) + " | "
            + str(row.get("title", ""))).lower()


def detect_task(meta_df):
    """自动识别可用的临床任务与标签"""
    tasks = {}
    if meta_df is None or not len(meta_df):
        return tasks
    txts = meta_df.apply(text_of, axis=1)

    # --- 结局（生存）---
    t = txts.str.extract(r"(surviv\w*|died|death|non-?survivor|deceased|alive)", expand=False)
    if t.notna().sum() >= 20:
        lab = t.map(lambda v: "control" if (pd.notna(v) and ("surviv" in v or "alive" in v)) else
                    ("case" if pd.notna(v) else np.nan))
        if lab.dropna().nunique() == 2 and lab.value_counts().min() >= 8:
            tasks["outcome"] = lab

    # --- ER 状态 ---
    t = txts.str.extract(r"er[_ ]?status[^:]*:\s*([^|\s]+)", expand=False)
    if t.notna().sum() >= 20:
        lab = t.map(lambda v: "case" if re.search(r"pos|^p$|\+", str(v)) else
                    ("control" if re.search(r"neg|^n$|\-", str(v)) else np.nan))
        if lab.dropna().nunique() == 2 and lab.value_counts().min() >= 8:
            tasks["er_status"] = lab

    # --- 分期 ---
    t = txts.str.extract(r"(tnm[\.\s]?stage|stage)[^:]*:\s*([^|\s]*)", expand=False)[1]
    if t.notna().sum() >= 20:
        lab = t.map(lambda v: "case" if re.search(r"(iii|iv|3|4)", str(v)) else
                    ("control" if re.search(r"(i\b|ii\b|1|2)", str(v)) else np.nan))
        if lab.dropna().nunique() == 2 and lab.value_counts().min() >= 8:
            tasks["stage"] = lab

    # --- 组织类型（疾病 vs 正常）---
    t = txts.map(lambda s: ("control" if any(k in s for k in
                    ["normal", "healthy", "control", "adjacent", "non-tumor",
                     "nontumor", "unaffected", "benign"])
                    else ("case" if any(k in s for k in
                          ["tumor", "cancer", "carcinoma", "disease", "patient",
                           "adenoma", "malignant", "cirrhosis", "sepsis"]) else np.nan)))
    if t.notna().sum() >= 20 and t.dropna().nunique() == 2:
        vc = t.value_counts()
        if vc.min() >= 8:
            tasks["disease"] = t
    return tasks


def module_features(expr, lmap, genes):
    feats = []
    for m in sorted(set(lmap.values())):
        cols = [g for g in genes if lmap.get(g) == m]
        if len(cols) < 3:
            continue
        X = expr[cols].values
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        u, s, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        feats.append(u[:, 0] * s[0])
    return np.vstack(feats).T if feats else None


def druggable_enrichment(labels, genes, gene2terms, terms, universe):
    """可成药性富集：Hub 基因（模块内高连接度）是否富集在 drug binding 类功能"""
    labs = np.asarray(labels)
    tom_genes = np.array(genes)
    hub_terms = [t for t in terms if re.search(r"drug bind|enzyme bind|ATP bind|"
                                               r"receptor activ|kinase activ", t, re.I)]
    if not hub_terms:
        return np.nan
    drug_genes = set().union(*[terms[t] for t in hub_terms])
    # 每个模块取内部度最高的基因作为 Hub
    hubs = []
    for m in sorted(set(labs[labs >= 0])):
        idx = np.where(labs == m)[0]
        if len(idx) < 3:
            continue
        hubs.append(genes[idx[0]])
    if len(hubs) < 3:
        return np.nan
    hits = sum(1 for h in hubs if h in drug_genes)
    # 超几何检验
    n = len(hubs)
    K = len(drug_genes & set(genes))
    N = len(genes)
    p = hypergeom.sf(hits - 1, N, K, n) if K > 0 else 1.0
    return float(-np.log10(max(p, 1e-300)))


def main(n_boot=8):
    meta_files = {f.stem.replace("_meta", ""): f for f in DATA.glob("*_meta.csv")}
    all_csv = sorted([f for f in DATA.glob("*.csv") if not f.name.endswith("_meta.csv")])
    terms, allg, gene2terms = L2.load_genesets()
    universe = len(allg)

    import gwas_full_analysis_v3 as V3
    go_sets = V3.load_go_sets()
    algos = {}
    algos.update(GA.ALGOS)
    algos.update(A2.NEW_ALGOS)
    for _a in ["Link Communities", "Edge Betweenness", "Optimal Modularity",
               "Markov Stability"]:
        algos.pop(_a, None)

    records = []
    t0 = time.time()
    for f in all_csv:
        gse = f.stem
        try:
            raw = pd.read_csv(f, index_col=0)
            if raw.shape[1] < 15 or raw.shape[0] < 1000:
                continue
        except Exception:
            continue
        raw.index = [str(i).strip().strip('"').strip().upper() for i in raw.index]
        raw = raw[~raw.index.duplicated()]
        import re as _re
        keep_mask = [_re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(i)) is not None
                     for i in raw.index]
        if sum(keep_mask) < 500:
            print(f"  ⏭ {gse} 行名非基因符号，跳过")
            continue
        raw = raw[keep_mask]
        import gwas_full_analysis_v3 as _V3
        df = _V3.preprocess_robust(raw)
        df.columns = [str(c).strip().strip('"') for c in df.columns]

        meta_df = None
        if gse in meta_files:
            try:
                meta_df = pd.read_csv(meta_files[gse])
            except Exception:
                meta_df = None
        tasks = detect_task(meta_df)
        # 关键修复：把标签索引绑定为 GSM（与表达矩阵列名对齐）
        if tasks and meta_df is not None and "gsm" in meta_df.columns:
            gsm_idx = meta_df["gsm"].astype(str).str.strip().str.strip('"').values
            tasks = {k: pd.Series(v.values, index=gsm_idx[:len(v)])
                     for k, v in tasks.items()}
        if not tasks:
            log_skip = f"{gse}: 无可用临床标签"
            print(f"  ⏭ {log_skip}")
            continue
        print(f"  {gse}: 任务 {list(tasks.keys())}")

        paths = V3.select_pathways(go_sets, set(df.index))
        for pw_name, pw_genes in paths[:4]:
            avail = [g for g in pw_genes if g in df.index]
            if len(avail) < 25:
                continue
            df_sub = df.loc[avail]
            tom = GA.compute_tom(df_sub)
            g = GA.to_ig(tom, list(df_sub.index))
            for aname, afn in algos.items():
                try:
                    labels = np.asarray(afn(g))
                except Exception:
                    continue
                lmap = {gene: int(labels[i]) for i, gene in enumerate(df_sub.index)}
                # 可成药性富集
                drg = druggable_enrichment(labels, list(df_sub.index), gene2terms,
                                           terms, universe)
                # 各任务的分类性能
                for tname, lab in tasks.items():
                    lab = lab.copy()
                    lab.index = [str(i).upper() for i in lab.index]
                    idx_ok = [s for s in df_sub.columns if s in lab.index
                              and lab.loc[s] in ("case", "control")]
                    if len(idx_ok) < 20:
                        continue
                    y = np.array([1 if lab.loc[s] == "case" else 0 for s in idx_ok])
                    if min(np.bincount(y)) < 6:
                        continue
                    X = module_features(df_sub[idx_ok].T, lmap, list(df_sub.index))
                    if X is None or X.shape[1] < 2:
                        continue
                    try:
                        clf = LogisticRegression(max_iter=800, C=0.5)
                        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
                        auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()
                    except Exception:
                        continue
                    records.append({"gse": gse, "task": tname, "pathway": pw_name,
                                    "algo": aname.replace("\n", " "),
                                    "n_samples": len(idx_ok),
                                    "n_modules": X.shape[1], "AUC": float(auc),
                                    "druggable_purity": drg})
        print(f"    累计 {len(records)} 条 ({time.time()-t0:.0f}s)")

    df_res = pd.DataFrame(records)
    df_res.to_csv(OUT / "downstream_v3.csv", index=False)
    print(f"\n✅ 已保存 downstream_v3.csv ({len(df_res)} 行)")
    if len(df_res):
        print(df_res.groupby(["task", "algo"])["AUC"].mean().unstack(0).round(3))
    return df_res


if __name__ == "__main__":
    main()
