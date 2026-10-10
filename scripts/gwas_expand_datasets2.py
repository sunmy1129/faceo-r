#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版数据集处理：修正 GPL 注释 URL + 稳健解析 + 探针->基因映射
GPL URL 规则：GPL + 数字前3位 + nnn（如 GPL570 -> GPL570nnn）
"""
import re
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
ANNOT = Path("/Users/sunmy/.hermes/gwas_data/annot")
ANNOT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    # 非癌症
    "GSE33000",   # 阿尔茨海默 (脑)
    "GSE36961",   # 心肌病 (心)
    "GSE72509",   # 系统性红斑狼疮
    "GSE61635",   # 系统性红斑狼疮
    "GSE15653",   # 2型糖尿病/肥胖 (肝)
    "GSE20966",   # 2型糖尿病 (胰岛)
    "GSE89408",   # 类风湿关节炎
    # 癌症扩充
    "GSE42568",   # 乳腺癌
    "GSE17536",   # 结肠癌 (预后)
    "GSE14520",   # 肝癌
    "GSE10072",   # 肺癌
    "GSE19188",   # 肺癌
    "GSE19804",   # 肺癌
    "GSE31210",   # 肺癌
    "GSE68465",   # 肺癌 (预后)
    "GSE32646",   # 乳腺癌
    "GSE22820",   # 乳腺癌
    "GSE17907",   # 乳腺癌
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(url, dest, timeout=300):
    if dest.exists() and dest.stat().st_size > 500:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def gpl_dir(gpl):
    """GPL570 -> GPL570nnn ; GPL96 -> GPL96nnn"""
    digits = re.sub(r"\D", "", gpl)
    return f"GPL{digits[:3]}nnn"


def get_gpl_annot(gpl):
    """下载并解析 GPL 注释表 -> {probe: gene_symbol}"""
    dest = ANNOT / f"{gpl}.annot.gz"
    urls = [
        f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{gpl_dir(gpl)}/{gpl}/annot/{gpl}.annot.gz",
        f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{gpl_dir(gpl)}/{gpl}/soft/{gpl}_family.soft.gz",
    ]
    for url in urls:
        try:
            fetch(url, dest, timeout=300)
            break
        except Exception:
            continue
    if not dest.exists():
        return None
    mapping = {}
    try:
        with gzip.open(dest, "rt", encoding="utf-8", errors="ignore") as f:
            header = None
            sym_col = None
            for line in f:
                if line.startswith("#") or line.startswith("^"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if header is None:
                    header = [p.strip().strip('"').lower() for p in parts]
                    for cand in ("gene symbol", "genesymbol", "symbol", "gene_symbol",
                                 "gene assignment", "gene names"):
                        if cand in header:
                            sym_col = header.index(cand)
                            break
                    if sym_col is None:
                        sym_col = 1
                    continue
                if len(parts) <= sym_col:
                    continue
                probe = parts[0].strip().strip('"')
                sym = parts[sym_col].strip().strip('"')
                if sym:
                    m = re.search(r"([A-Za-z0-9\-\.]+)", sym.split("///")[0])
                    if m:
                        mapping[probe] = m.group(1).upper()
    except Exception as e:
        log(f"    annot 解析失败 {gpl}: {str(e)[:50]}")
        return None
    return mapping or None


def parse_matrix(mf):
    """稳健解析 series matrix"""
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
                key = key.lstrip("!")
                vals = [v.strip().strip('"') for v in val.strip().split("\t")]
                meta.setdefault(key, []).extend(vals)
    if len(table_lines) < 3:
        return None, None
    # 第一行是表头（ID_REF + 样本），数据行可能含引号
    df = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t",
                     index_col=0, quoting=3, low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    if df.shape[0] == 0:
        return None, None
    ids = meta.get("Sample_geo_accession", [])
    sm = pd.DataFrame({"gsm": ids if len(ids) == df.shape[1] else list(df.columns)})
    sm["title"] = meta.get("Sample_title", [""] * len(sm))[:len(sm)]
    sm["source"] = meta.get("Sample_source_name_ch1", [""] * len(sm))[:len(sm)]
    chars = [meta[k][:len(sm)] for k in meta if k.startswith("Sample_characteristics_ch1")]
    sm["chars"] = [" | ".join(x) for x in zip(*chars)] if chars else ""
    # GPL
    gpl = None
    for k in ("Series_platform_id", "Sample_platform_id"):
        if k in meta and meta[k]:
            gpl = meta[k][0].strip()
            break
    sm.attrs["gpl"] = gpl
    return df, sm


def process(gse):
    mf = RAW / f"{gse}_series_matrix.txt.gz"
    if not mf.exists():
        pre = re.sub(r"\d{3}$", "nnn", gse)
        url = (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{pre}/{gse}/matrix/"
               f"{gse}_series_matrix.txt.gz")
        try:
            fetch(url, mf)
        except Exception as e:
            log(f"  ✗ {gse} 下载失败: {str(e)[:50]}")
            return None
    df, sm = parse_matrix(mf)
    if df is None or df.shape[1] < 10:
        log(f"  ✗ {gse} 解析失败")
        return None
    gpl = sm.attrs.get("gpl")
    log(f"  {gse}: {df.shape[0]} 探针 × {df.shape[1]} 样本 | 平台 {gpl}")
    mapping = get_gpl_annot(gpl) if gpl else None
    if mapping:
        idx = pd.Index([mapping.get(str(p), "") for p in df.index])
        df.index = idx
        d2 = df[df.index != ""]
        d2 = d2.groupby(level=0).mean()
        log(f"    注释成功: -> {d2.shape[0]} 基因")
        out = d2
    else:
        log(f"    ⚠ 注释失败，保留探针")
        out = df
    out.to_csv(PROC / f"{gse}.csv")
    sm.drop(columns=[], errors="ignore").to_csv(PROC / f"{gse}_meta.csv", index=False)
    return {"gse": gse, "probes_or_genes": out.shape[0], "samples": out.shape[1],
            "annotated": bool(mapping), "gpl": gpl}


if __name__ == "__main__":
    res = []
    for gse in TARGETS:
        log(f"处理 {gse}")
        try:
            r = process(gse)
            if r:
                res.append(r)
                log(f"  ✅ {gse}: {r['probes_or_genes']} × {r['samples']} (注释={r['annotated']})")
        except Exception as e:
            log(f"  ✗ {gse}: {str(e)[:70]}")
    pd.DataFrame(res).to_csv(RAW / "process_summary.csv", index=False)
    log(f"完成 {len(res)}/{len(TARGETS)}")
