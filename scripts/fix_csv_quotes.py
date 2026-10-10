#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复 CSV 引号格式（列名/行名的三重引号）"""
import csv
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
PROC = Path("/Users/sunmy/Documents/QF/GWAS/qcdata")


def clean(s):
    return str(s).strip().strip('"').strip()


def fix(f):
    df = pd.read_csv(f, index_col=0, quoting=3, low_memory=False)
    df.index = [clean(i) for i in df.index]
    df.columns = [clean(c) for c in df.columns]
    df = df[~df.index.duplicated()]
    df.to_csv(f, quoting=csv.QUOTE_NONE)
    return df.shape


if __name__ == "__main__":
    fixed = []
    for f in sorted(PROC.glob("*.csv")):
        if f.name.endswith("_meta.csv"):
            continue
        try:
            h = open(f, "rb").read(80).decode("utf-8", "ignore")
            if '"""' in h or f.stat().st_size < 10000:
                shape = fix(f)
                fixed.append((f.stem, shape))
                print(f"  ✅ 修复 {f.stem}: {shape}")
        except Exception as e:
            print(f"  ✗ {f.stem}: {str(e)[:60]}")
    print(f"\n共修复 {len(fixed)} 个文件")
