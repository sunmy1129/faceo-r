#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L6 下游任务验证：用划分模块的特征基因预测临床表型
- 特征：每个模块的第一主成分（module eigengene）
- 任务：疾病状态 / 分子亚型 / 分期 分类（5 折交叉验证 AUC）
- 目的：检验「精细拆分」是否带来实际预测增益
输出：downstream_results.csv
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

import gwas_full_analysis as GA

warnings.filterwarnings("ignore")
BASE = Path("/Users/sunmy/Documents/QF/GWAS")
HUB, DATA = BASE / "output/hubs", BASE / "qcdata"
META = Path("/Users/sunmy/.hermes/gwas_data")
OUT = BASE / "output/core_results"
SEED = 42

# 每个数据集采用的临床标签与提取规则
LABEL_RULES = {
    "GSE137342": ("outcome", lambda txt: (
        "case" if "died" in txt.lower() or "non-survivor" in txt.lower() or "nonsurvivor" in txt.lower()
        else ("control" if "survived" in txt.lower() else None))),
    "GSE25066": ("er_status", lambda txt: (
        "case" if re.search(r"er_status_ihc:\s*(p|positive)", txt.lower()) else
        ("control" if re.search(r"er_status_ihc:\s*(n|negative)", txt.lower()) else None))),
    "GSE39582": ("stage", lambda txt: (
        "case" if re.search(r"tnm\.stage:\s*(3|4)", txt.lower()) else
        ("control" if re.search(r"tnm\.stage:\s*(1|2)", txt.lower()) else None))),
    "GSE6764": ("tissue", lambda txt: (
        "control" if any(k in txt.lower() for k in ["cirrhotic", "cirrhosis", "normal", "non-tum"])
        else ("case" if "tumor" in txt.lower() or "carcinoma" in txt.lower() else None))),
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_labels(gse):
    f = META / f"{gse}_samples.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    field, rule = LABEL_RULES.get(gse, (None, None))
    if rule is None:
        return None
    df["label"] = df["chars"].fillna("").apply(rule)
    if df["label"].isna().any():
        df["label"] = df["label"].fillna(
            df["source"].fillna("").apply(rule))
    d = df.dropna(subset=["label"])
    return d.set_index("gsm")["label"]


def module_features(expr, labels_map, genes):
    """
    expr: 样本 × 基因（已对齐）
    labels_map: 基因 -> 模块 ID
    返回：样本 × 模块特征（第一主成分）
    """
    feats = []
    for m in sorted(set(labels_map.values())):
        cols = [g for g in genes if labels_map.get(g) == m]
        if len(cols) < 3:
            continue
        X = expr[cols].values
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        u, s, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        feats.append(u[:, 0] * s[0])
    if not feats:
        return None
    return np.vstack(feats).T


def main():
    metas = sorted(HUB.glob("*_meta.json"))
    rows = []
    t0 = time.time()
    cache = {}
    for mi, mf in enumerate(metas, 1):
        meta = json.load(open(mf))
        gse, pw = meta["gse"], meta["pathway"]
        if gse not in cache:
            lab = load_labels(gse)
            if lab is None or len(lab) < 20:
                cache[gse] = (None, None, None)
                log(f"  {gse}: 标签不足，跳过 L6")
                continue
            df = GA.preprocess(pd.read_csv(DATA / f"{gse}.csv", index_col=0))
            cache[gse] = (df, lab, None)
        df, lab, _ = cache[gse]
        if df is None:
            continue
        # 对齐样本
        common_samples = [s for s in df.columns if s in lab.index]
        if len(common_samples) < 20:
            continue
        y = lab.loc[common_samples].map({"case": 1, "control": 0}).values
        if len(set(y)) < 2 or min(np.bincount(y)) < 8:
            continue

        avail = [g for g in meta["genes"] if g in df.index]
        if len(avail) < 20:
            continue
        df_sub = df.loc[avail]
        tom = GA.compute_tom(df_sub)
        g = GA.to_ig(tom, list(df_sub.index))

        for aname, afn in GA.ALGOS.items():
            try:
                labels = np.asarray(afn(g))
            except Exception:
                continue
            lmap = {gene: int(labels[i]) for i, gene in enumerate(df_sub.index)}
            X = module_features(df_sub[common_samples].T, lmap, list(df_sub.index))
            if X is None or X.shape[1] < 2:
                continue
            # 5 折交叉验证 AUC
            try:
                clf = LogisticRegression(max_iter=800, C=0.5)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()
            except Exception:
                continue
            rows.append({"gse": gse, "pathway": pw, "algo": aname.replace("\n", " "),
                         "task": LABEL_RULES[gse][0], "n_samples": len(common_samples),
                         "n_modules_used": X.shape[1], "AUC": float(auc)})
        if mi % 6 == 0:
            log(f"({mi}/{len(metas)}) 累计 {len(rows)} 条")

    df_res = pd.DataFrame(rows)
    df_res.to_csv(OUT / "downstream_results.csv", index=False)
    log(f"✅ 已保存 downstream_results.csv ({len(df_res)} 行, {time.time()-t0:.0f}s)")

    if len(df_res):
        pd.set_option("display.width", 200)
        print("\n" + "="*100)
        print("L6 下游任务：模块特征预测临床表型（5 折 CV AUC）")
        print("="*100)
        piv = df_res.groupby(["task", "algo"])["AUC"].mean().unstack(0).round(3)
        print(piv.sort_values(piv.columns[0], ascending=False))
        print()
        print("按算法汇总：")
        print(df_res.groupby("algo").agg(AUC均值=("AUC", "mean"),
                                        任务数=("AUC", "count")).round(3)
              .sort_values("AUC均值", ascending=False))
    return df_res


if __name__ == "__main__":
    main()
