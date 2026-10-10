#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集处理 v3（最终版）
- 注释：mygene.info 在线 API（探针 -> 基因符号）
- 内存：float32 + 高变异基因过滤 + 及时释放
- 稳健：逐数据集容错，跳过失败者
"""
import re
import gc
import gzip
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import io

warnings.filterwarnings("ignore")
RAW = Path("/Users/sunmy/.hermes/gwas_data/new_raw")
PROC = Path("/Users/sunmy/Documents/QF/GWAS/qcdata")
RAW.mkdir(parents=True, exist_ok=True)
MAX_GENES = 15000   # 每数据集保留的高变异基因上限

TARGETS = [
    "GSE42568", "GSE17536", "GSE14520", "GSE10072", "GSE19188",
    "GSE19804", "GSE31210", "GSE68465", "GSE32646", "GSE22820",
    "GSE17907", "GSE36961", "GSE72509", "GSE61635", "GSE15653",
    "GSE20966",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(url, dest, timeout=300):
    if dest.exists() and dest.stat().st_size > 500:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def parse_matrix(mf):
    meta, table_lines, in_table = {}, [], False
    with gzip.open(mf, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                table_lines.append(line.rstrip("\n"))
            elif line.startswith("!"):
                key, _, val = line.partition("\t")
                meta.setdefault(key.lstrip("!"), []).extend(
                    [v.strip().strip('"') for v in val.strip().split("\t")])
    if len(table_lines) < 3:
        return None, None
    df = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", index_col=0,
                     quoting=3, low_memory=False)
    # 去掉探针 ID 的引号（关键修复）
    df.index = [str(i).strip().strip('"') for i in df.index]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if df.shape[0] == 0 or df.shape[1] < 10:
        return None, None
    df = df.astype(np.float32)
    ids = meta.get("Sample_geo_accession", [])
    sm = pd.DataFrame({"gsm": ids if len(ids) == df.shape[1] else list(df.columns)})
    sm["title"] = meta.get("Sample_title", [""] * len(sm))[:len(sm)]
    sm["source"] = meta.get("Sample_source_name_ch1", [""] * len(sm))[:len(sm)]
    chars = [meta[k][:len(sm)] for k in meta if k.startswith("Sample_characteristics_ch1")]
    sm["chars"] = [" | ".join(x) for x in zip(*chars)] if chars else ""
    return df, sm


def annotate_mygene(probes):
    """用 mygene.info 注释探针 -> 基因符号"""
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        mapping = {}
        probes = [str(p) for p in probes]
        # 分批查询
        for i in range(0, len(probes), 900):
            batch = probes[i:i + 900]
            try:
                res = mg.querymany(batch, scopes="reporter,alias,accession",
                                   fields="symbol", species="human",
                                   verbose=False, returnall=False)
                for r in res:
                    if "symbol" in r and r["symbol"]:
                        q = r["query"]
                        if q not in mapping:
                            mapping[q] = str(r["symbol"]).upper()
            except Exception:
                continue
        return mapping or None
    except Exception:
        return None


def process(gse):
    mf = RAW / f"{gse}_series_matrix.txt.gz"
    flat = PROC / f"{gse}.csv"
    if flat.exists() and flat.stat().st_size > 200000:
        log(f"  {gse} 已处理，跳过")
        return {"gse": gse, "skipped": True}
    if not mf.exists():
        pre = re.sub(r"\d{3}$", "nnn", gse)
        try:
            fetch(f"https://ftp.ncbi.nlm.nih.gov/geo/series/{pre}/{gse}/matrix/"
                  f"{gse}_series_matrix.txt.gz", mf)
        except Exception as e:
            log(f"  ✗ {gse} 下载失败: {str(e)[:50]}")
            return None
    df, sm = parse_matrix(mf)
    if df is None:
        log(f"  ✗ {gse} 解析失败（可能无表达数据）")
        return None
    log(f"  {gse}: {df.shape[0]} 探针 × {df.shape[1]} 样本")
    # 高变异过滤（内存优化）
    if df.shape[0] > MAX_GENES:
        v = df.var(axis=1).fillna(0)
        keep = v.nlargest(MAX_GENES).index
        df = df.loc[keep]
        log(f"    高变异过滤 -> {df.shape[0]} 探针")
    # 注释
    mapping = annotate_mygene(df.index.tolist())
    if mapping:
        idx = pd.Index([mapping.get(str(p), "") for p in df.index])
        df.index = idx
        df = df[df.index != ""]
        df = df.groupby(level=0).mean()
        log(f"    mygene 注释 -> {df.shape[0]} 基因")
    else:
        log(f"    ⚠ 注释失败，跳过该数据集")
        return None
    df.to_csv(PROC / f"{gse}.csv")
    sm.to_csv(PROC / f"{gse}_meta.csv", index=False)
    del df, sm
    gc.collect()
    return {"gse": gse, "genes": None, "samples": None, "ok": True}


if __name__ == "__main__":
    res = []
    for gse in TARGETS:
        log(f"处理 {gse}")
        try:
            r = process(gse)
            if r:
                res.append(r)
        except Exception as e:
            log(f"  ✗ {gse} 异常: {str(e)[:70]}")
        gc.collect()
    log(f"完成 {len(res)}/{len(TARGETS)}")
