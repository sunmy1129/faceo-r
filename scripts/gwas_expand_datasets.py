#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集扩展：下载并预处理新 GEO 队列为统一的「基因 × 样本」矩阵
流程：series matrix -> 解析元数据与表达值 -> GPL 注释映射探针->基因 -> 聚合 -> 保存
"""
import re
import gzip
import time
import json
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
PROC.mkdir(parents=True, exist_ok=True)

# 新增队列：GSE ID -> (疾病, 组织, 分组关键词配置)
NEW_SETS = {
    "GSE89408":  ("Rheumatoid arthritis", "synovium"),
    "GSE33000":  ("Alzheimer's disease", "brain"),
    "GSE15653":  ("Type 2 diabetes / obesity", "liver"),
    "GSE36961":  ("Cardiomyopathy", "heart"),
    "GSE72509":  ("Systemic lupus erythematosus", "whole blood"),
    "GSE61635":  ("Systemic lupus erythematosus", "whole blood"),
    "GSE20966":  ("Type 2 diabetes", "pancreatic islet"),
    "GSE42568":  ("Breast cancer", "breast"),
    "GSE17536":  ("Colon cancer (prognosis)", "colon"),
    "GSE14520":  ("Hepatocellular carcinoma", "liver"),
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(url, dest, timeout=180):
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def download_matrix(gse):
    pre = re.sub(r"\d{3}$", "nnn", gse)
    url = (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{pre}/{gse}/matrix/"
           f"{gse}_series_matrix.txt.gz")
    dest = RAW / f"{gse}_series_matrix.txt.gz"
    try:
        fetch(url, dest)
        return dest
    except Exception as e:
        log(f"  ✗ {gse} matrix 下载失败: {str(e)[:60]}")
        return None


def parse_matrix(path):
    """解析 series matrix：返回 (表达矩阵 df, 样本元数据 df)"""
    meta, table_lines, in_table = {}, [], False
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
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
                key = key.lstrip("!")
                vals = [v.strip('"') for v in val.strip().split("\t")]
                meta.setdefault(key, []).extend(vals)
    if not table_lines:
        return None, None
    df = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", index_col=0)
    df.index.name = "probe"
    # 样本元数据
    ids = meta.get("Sample_geo_accession", [])
    n = len(df.columns)
    sm = pd.DataFrame({"gsm": ids[:n] if len(ids) >= n else list(df.columns)})
    sm["title"] = meta.get("Sample_title", [""] * n)[:n]
    sm["source"] = meta.get("Sample_source_name_ch1", [""] * n)[:n]
    chars = []
    for k in meta:
        if k.startswith("Sample_characteristics_ch1"):
            chars.append(meta[k][:n])
    sm["chars"] = [" | ".join(x) for x in zip(*chars)] if chars else ""
    return df, sm


def download_gpl(gse):
    """获取平台注释（探针->基因符号）"""
    pre = re.sub(r"\d{3}$", "nnn", gse)
    # 从 matrix 里找出平台
    return None


def annotate(df, gpl_id):
    """用 GPL 注释把探针映射为基因符号"""
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{gpl_id[:5]}nnn/{gpl_id}/annot/{gpl_id}.annot.gz"
    dest = RAW / f"{gpl_id}.annot.gz"
    try:
        fetch(url, dest, timeout=240)
    except Exception:
        return None
    mapping = {}
    try:
        with gzip.open(dest, "rt", encoding="utf-8", errors="ignore") as f:
            header = None
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if header is None:
                    header = parts
                    continue
                if len(parts) < 2:
                    continue
                probe, sym = parts[0], parts[1]
                if sym and sym != "":
                    mapping[probe] = sym.split("///")[0].strip()
    except Exception:
        return None
    return mapping or None


def process(gse):
    disease, tissue = NEW_SETS[gse]
    mf = download_matrix(gse)
    if mf is None:
        return None
    try:
        df, sm = parse_matrix(mf)
    except Exception as e:
        log(f"  ✗ {gse} 解析失败: {str(e)[:60]}")
        return None
    if df is None or df.shape[1] < 10:
        log(f"  ✗ {gse} 数据不足 ({df.shape if df is not None else None})")
        return None
    n_probe, n_sample = df.shape
    log(f"  {gse}: {n_probe} 探针 × {n_sample} 样本")

    # 平台注释
    pre = re.sub(r"\d{3}$", "nnn", gse)
    plat_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{pre}/{gse}/matrix/"
    # 从 matrix 头部找 GPL
    gpl = None
    try:
        with gzip.open(mf, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("!Series_platform_id"):
                    gpl = line.strip().split("\t")[-1].strip('"')
                    break
                if line.startswith("!series_matrix_table_begin"):
                    break
    except Exception:
        pass

    mapping = annotate(df, gpl) if gpl else None
    if mapping:
        idx = pd.Index([mapping.get(p, "") for p in df.index])
        df2 = df.copy()
        df2.index = idx
        df2 = df2[df2.index != ""]
        df2 = df2.groupby(level=0).mean()
        log(f"    注释映射: {n_probe} 探针 -> {df2.shape[0]} 基因")
        out_df = df2
    else:
        log(f"    ⚠ 无注释，保留探针级数据")
        out_df = df

    out_df.to_csv(PROC / f"{gse}.csv")
    sm.to_csv(PROC / f"{gse}_meta.csv", index=False)
    return {"gse": gse, "disease": disease, "tissue": tissue,
            "genes": out_df.shape[0], "samples": out_df.shape[1],
            "annotated": bool(mapping), "gpl": gpl}


if __name__ == "__main__":
    results = []
    for gse in NEW_SETS:
        log(f"处理 {gse} ({NEW_SETS[gse][0]})")
        try:
            r = process(gse)
            if r:
                results.append(r)
                log(f"  ✅ {gse}: {r['genes']} 基因 × {r['samples']} 样本")
        except Exception as e:
            log(f"  ✗ {gse}: {str(e)[:80]}")
    pd.DataFrame(results).to_csv(RAW / "download_summary.csv", index=False)
    log(f"完成：成功 {len(results)}/{len(NEW_SETS)}")
