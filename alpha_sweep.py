#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""α 精细扫描：找最优 α，为重新设计自适应区间提供依据"""
import sys, time, re
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import numpy as np
import pandas as pd
from multiprocessing import Pool

import faceo_enhanced as FE
import gwas_faceo_r as FR
import gwas_full_analysis as GA
import gwas_full_analysis_v3 as V3

BASE, OUT, GS = FE.BASE, FE.OUT, FE.GS
ALPHAS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]


def work(job):
    gse, pw_name, pw_genes, kind, lvl = job
    out = []
    try:
        raw = pd.read_csv(BASE / f"qcdata/{gse}.csv", index_col=0,
                          quoting=3, low_memory=False)
        raw.index = [str(i).strip().strip('"').upper() for i in raw.index]
        keep = [re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(i)) is not None
                for i in raw.index]
        df_full = V3.preprocess_robust(raw[keep])
    except Exception:
        return out
    genes = [g for g in pw_genes if g in df_full.index]
    if not (60 <= len(genes) <= 200):
        return out
    genes = list(genes)
    df = FE.degrade(df_full.loc[genes], kind, lvl)
    try:
        tom = GA.compute_tom(df)
        g = GA.to_ig(tom, genes)
    except Exception:
        return out
    g2t, re2t = FE._G["go2t"], FE._G["re2t"]
    q = FE.reliability_of(tom, genes) if hasattr(FE, "reliability_of") else 0.5
    try:
        ld = GA.leiden_simple(g, resolution=1.0) if hasattr(GA, "leiden_simple") else None
    except Exception:
        ld = None
    for a in ALPHAS:
        try:
            mods = FE.faceo_partition(tom, genes, g2t, alpha=a) if hasattr(FE, "faceo_partition") else None
        except Exception:
            mods = None
    return out


if __name__ == "__main__":
    print("使用 faceo_enhanced 的内部函数进行 α 扫描")
