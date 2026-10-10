#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L2 功能层评估（修正版，无大小偏差）
主指标：模块功能内聚性 z 分数 —— 模块内基因对的 GO 注释 Jaccard 相似度均值，
        相对同一通路内随机基因集的期望做标准化（消除通路本身的富集背景）
辅助指标：富集纯度（限制基因集大小 ≥10，并对 p 值上限截断）
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import hypergeom

GS_DIR = Path("/Users/sunmy/.hermes/gwas_data")
_CACHE = {}
MIN_TERM_SIZE = 10
P_CAP = 30.0  # -log10(p) 上限截断，防止极小基因集导致虚高


def load_genesets(libs=("GO_Biological_Process_2021", "GO_Molecular_Function_2021",
                        "GO_Cellular_Component_2021", "KEGG_2021_Human", "Reactome_2022")):
    key = tuple(libs)
    if key in _CACHE:
        return _CACHE[key]
    terms, allg = {}, set()
    for lib in libs:
        f = GS_DIR / f"{lib}.json"
        if not f.exists():
            continue
        for name, genes in json.load(open(f)).items():
            gs = frozenset(g.strip().upper() for g in genes)
            if MIN_TERM_SIZE <= len(gs) <= 1000:
                terms[name] = gs
                allg |= gs
    # 基因 -> term 集合
    gene2terms = defaultdict(set)
    for name, gs in terms.items():
        for g in gs:
            gene2terms[g].add(name)
    _CACHE[key] = (terms, allg, dict(gene2terms))
    return _CACHE[key]


def coherence(module_genes, gene2terms, min_annot=3, max_genes=60):
    """模块内基因对的功能相似度均值（Jaccard）；大模块采样以控制复杂度"""
    sets = [gene2terms.get(g.upper()) for g in module_genes]
    sets = [s for s in sets if s]
    if len(sets) < min_annot:
        return np.nan
    if len(sets) > max_genes:
        idx = np.random.default_rng(0).choice(len(sets), max_genes, replace=False)
        sets = [sets[i] for i in idx]
    sims = []
    for i in range(len(sets)):
        si = sets[i]
        for j in range(i + 1, len(sets)):
            sj = sets[j]
            u = len(si | sj)
            if u:
                sims.append(len(si & sj) / u)
    return float(np.mean(sims)) if sims else np.nan


def coherence_z(module_genes, gene2terms, pool, n_perm=15, seed=42):
    """相对通路内随机基因集的标准化 z 分数（预计算注释集合加速）"""
    obs = coherence(module_genes, gene2terms)
    if obs != obs:
        return np.nan, np.nan
    m = min(len(module_genes), 60)
    pool_sets = [gene2terms.get(g.upper()) for g in pool]
    pool_sets = [s for s in pool_sets if s]
    if len(pool_sets) < m:
        return np.nan, obs
    rng = np.random.default_rng(seed)

    def coh_of(sets):
        sims = []
        for i in range(len(sets)):
            si = sets[i]
            for j in range(i + 1, len(sets)):
                u = len(si | sets[j])
                if u:
                    sims.append(len(si & sets[j]) / u)
        return float(np.mean(sims)) if sims else np.nan

    null = []
    for _ in range(n_perm):
        sel = rng.choice(len(pool_sets), m, replace=False)
        v = coh_of([pool_sets[i] for i in sel])
        if v == v:
            null.append(v)
    if len(null) < 5:
        return np.nan, obs
    null = np.asarray(null)
    return float((obs - null.mean()) / (null.std() + 1e-9)), obs


def enrichment_purity(module_genes, terms, universe, min_overlap=2):
    """辅助指标：富集纯度（BH 校正，p 值上限截断）"""
    M = set(g.upper() for g in module_genes)
    m = len(M)
    if m < 3:
        return np.nan
    pvals = []
    for name, gs in terms.items():
        inter = len(M & gs)
        if inter < min_overlap:
            continue
        pvals.append(hypergeom.sf(inter - 1, universe, len(gs), m))
    if not pvals:
        return np.nan
    pvals = np.sort(np.asarray(pvals))
    n = len(pvals)
    adj = pvals * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    best = float(np.clip(-np.log10(max(adj.min(), 1e-300)), 0, P_CAP))
    return best


def evaluate_functional(labels, genes, terms, universe, gene2terms, n_perm=60):
    """
    返回：
      coh_mean     模块功能内聚性均值
      coh_z_mean   模块内聚性 z 分数均值（主指标）
      purity_mean  富集纯度均值（辅助，已截断）
      n_eval       参与评估的模块数
    """
    labs = np.asarray(labels)
    valid = labs >= 0
    uniq = sorted(set(labs[valid]))
    cohs, zs, purs = [], [], []
    for u in uniq:
        members = [genes[i] for i in range(len(genes)) if labs[i] == u]
        if len(members) < 3:
            continue
        c = coherence(members, gene2terms)
        if c == c:
            cohs.append(c)
        z, _ = coherence_z(members, gene2terms, genes, n_perm=n_perm)
        if z == z:
            zs.append(z)
        p = enrichment_purity(members, terms, universe)
        if p == p:
            purs.append(p)
    def m_(x):
        return float(np.mean(x)) if x else np.nan
    return {
        "coh_mean": m_(cohs),
        "coh_z_mean": m_(zs),          # 主指标
        "purity_mean": m_(purs),       # 辅助
        "n_eval": len(cohs),
        "coverage": float(valid.mean()),
    }


if __name__ == "__main__":
    terms, allg, g2t = load_genesets()
    print(f"基因集库: {len(terms)} terms (min size {MIN_TERM_SIZE}), universe {len(allg)}, "
          f"有注释基因 {len(g2t)}")
    # 自检1：真凋亡模块 vs 随机模块
    apop = ["BAX", "BCL2", "CASP3", "CASP8", "CASP9", "BID", "BAK1", "APAF1",
            "CYCS", "DIABLO", "XIAP", "BIRC5", "FAS", "FADD", "TP53", "PMAIP1"]
    pool = apop + ["GAPDH", "ACTB", "TUBB", "ALB", "INS", "MYH7", "COL1A1",
                   "KRT14", "SFTPA1", "CD19", "MS4A1", "HBB", "FGA", "APOE"]
    z, obs = coherence_z(apop[:10], g2t, pool)
    zr, obsr = coherence_z(["GAPDH", "ACTB", "TUBB", "ALB", "INS"], g2t, pool)
    print(f"凋亡模块内聚性 z = {z:.2f} (obs {obs:.4f})")
    print(f"随机模块内聚性 z = {zr:.2f} (obs {obsr:.4f})")
    print("✅ 自检通过：功能模块 z 显著高于随机" if (z == z and z > 2) else "⚠️ 指标异常")
