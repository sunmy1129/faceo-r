#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 4：下载并处理 6 个新非癌症队列（复用已验证的解析逻辑）
复用 gwas_expand_datasets3 的 fetch/parse/preprocess 流程
"""
import gzip
import re
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
import urllib.request

warnings.filterwarnings("ignore")
BASE = Path("/Users/sunmy/Documents/QF/GWAS")
RAW = Path("/Users/sunmy/.hermes/gwas_data/new_raw")
PROC = BASE / "qcdata"
RAW.mkdir(parents=True, exist_ok=True)

NEW = [("GSE38642", "diabetes"), ("GSE55098", "metabolic"),
       ("GSE30153", "infection"), ("GSE39088", "autoimmune"),
       ("GSE21942", "autoimmune"), ("GSE26927", "renal")]

import mygene
mg = mygene.MyGeneInfo()


def fetch(url, dest, timeout=400):
    if dest.exists() and dest.stat().st_size > 500:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        while True:
            c = r.read(1 << 20)
            if not c:
                break
            f.write(c)
    return dest


def parse_matrix(mf):
    meta, tl, in_t = {}, [], False
    with gzip.open(mf, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_t = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_t = False
                continue
            if in_t:
                tl.append(line.rstrip("\n").split("\t"))
            elif line.startswith("!Sample_title"):
                meta["titles"] = [x.strip('"') for x in line.rstrip().split("\t")[1:]]
    if len(tl) < 3:
        return None, meta
    hdr = [x.strip('"') for x in tl[0]]
    rows = []
    for r in tl[1:]:
        if len(r) == len(hdr):
            rows.append(r)
    if not rows:
        return None, meta
    df = pd.DataFrame([r[1:] for r in rows],
                      index=[r[0].strip('"') for r in rows],
                      columns=hdr[1:])
    return df.astype(float), meta


def main():
    ok, fail = [], []
    for gse, cat in NEW:
        mf = RAW / f"{gse}_series_matrix.txt.gz"
        flat = PROC / f"{gse}.csv"
        if flat.exists() and flat.stat().st_size > 200000:
            print(f"  {gse} 已存在，跳过"); ok.append(gse); continue
        pre = re.sub(r"\d{3}$", "nnn", gse)
        try:
            fetch(f"https://ftp.ncbi.nlm.nih.gov/geo/series/{pre}/{gse}/matrix/"
                  f"{gse}_series_matrix.txt.gz", mf)
        except Exception as e:
            print(f"  ✗ {gse} 下载失败: {str(e)[:50]}"); fail.append(gse); continue
        df, meta = parse_matrix(mf)
        if df is None:
            print(f"  ✗ {gse} 解析失败（无表达矩阵）"); fail.append(gse); continue
        probes = [str(i).strip('"') for i in df.index]
        # 若为探针 ID，用 mygene 注释
        is_symbol = sum(1 for p in probes[:200]
                        if re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", p)) / min(200, len(probes))
        if is_symbol < 0.7:
            try:
                res = mg.querymany(probes, scopes="reporter", fields="symbol",
                                   species="human", verbose=False)
                m = {r["query"]: r["symbol"] for r in res
                     if "symbol" in r and r.get("notfound") is not True}
                df = df.loc[[p for p in df.index
                             if str(p).strip('"') in m]].copy()
                df.index = [m[str(p).strip('"')].upper() for p in df.index]
            except Exception as e:
                print(f"  ✗ {gse} 注释失败: {str(e)[:40]}"); fail.append(gse); continue
        else:
            df.index = [str(i).strip('"').upper() for i in df.index]
        df = df[~df.index.duplicated()]
        df = df[df.index.map(lambda x: bool(re.match(r"^[A-Z][A-Z0-9\-\.]{1,14}$", str(x))))]
        if df.shape[0] < 2000:
            print(f"  ✗ {gse} 基因数不足 ({df.shape[0]})"); fail.append(gse); continue
        if df.values.max() > 100:
            df = np.log2(df + 1)
        df.to_csv(flat)
        print(f"  ✅ {gse} ({cat}): {df.shape[0]} 基因 x {df.shape[1]} 样本")
        ok.append(gse)
    print(f"\n成功 {len(ok)}/{len(NEW)}: {ok}")
    print(f"失败: {fail}")


if __name__ == "__main__":
    main()
