#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补注释：将探针级 CSV 转为基因符号级"""
import sys
import warnings
import pandas as pd
import mygene
from pathlib import Path

warnings.filterwarnings("ignore")
PROC = Path("/Users/sunmy/Documents/QF/GWAS/qcdata")


def annotate(gse):
    f = PROC / f"{gse}.csv"
    df = pd.read_csv(f, index_col=0, quoting=3, low_memory=False)
    df.index = [str(i).strip().strip('"') for i in df.index]
    # 已是基因符号则跳过
    if all(str(i)[0].isalpha() and "_at" not in str(i) for i in df.index[:50]):
        print(f"  {gse} 已是基因符号，跳过")
        return None
    mg = mygene.MyGeneInfo()
    probes = [str(p) for p in df.index]
    mapping = {}
    for i in range(0, len(probes), 800):
        batch = probes[i:i + 800]
        try:
            res = mg.querymany(batch, scopes="reporter", fields="symbol",
                               species="human", verbose=False)
            for r in res:
                if r.get("symbol"):
                    q = r["query"]
                    if q not in mapping:
                        mapping[q] = str(r["symbol"]).upper()
        except Exception as e:
            print(f"    批 {i} 失败: {str(e)[:50]}")
    if not mapping:
        print(f"  ✗ {gse} 注释失败")
        return None
    idx = pd.Index([mapping.get(str(p), "") for p in df.index])
    df.index = idx
    df = df[df.index != ""]
    df = df.groupby(level=0).mean()
    df.to_csv(f, quoting=3)
    print(f"  ✅ {gse}: {len(mapping)} 探针 -> {df.shape[0]} 基因")
    return df.shape


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["GSE15653", "GSE61635"]
    for gse in targets:
        print(f"处理 {gse}")
        annotate(gse)
