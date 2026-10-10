#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 GEO 文本接口获取样本分组（用于 L6 下游任务）"""
import re
import time
import warnings
import pandas as pd
from pathlib import Path
import urllib.request

warnings.filterwarnings("ignore")
CACHE = Path("/Users/sunmy/.hermes/gwas_data")
CACHE.mkdir(exist_ok=True)
GSES = ["GSE137342", "GSE25066", "GSE39582", "GSE6764"]

# 手工校正的分组词表（依据各数据集的实验设计）
NEG = ["normal", "healthy", "control", "adjacent", "non-tumor", "nontumor",
       "non tumor", "unaffected", "benign", "no tumor", "cirrhosis", "fibrosis",
       "wild type", "wt", "sham", "vehicle", "untreated", "baseline"]
POS = ["tumor", "tumour", "cancer", "carcinoma", "hcc", "adenoma", "malignant",
       "sepsis", "patient", "disease", "lps", "treated", "dysplastic", "cll"]


def classify(text):
    t = str(text).lower()
    for k in NEG:
        if k in t:
            return "control"
    for k in POS:
        if k in t:
            return "case"
    return "unknown"


def fetch(gse, retries=2):
    out = CACHE / f"{gse}_samples.csv"
    if out.exists():
        return pd.read_csv(out)
    url = (f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}"
           f"&targ=gsm&form=text&view=brief")
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            txt = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
            break
        except Exception as e:
            print(f"  {gse} 尝试 {attempt+1} 失败: {str(e)[:60]}")
            time.sleep(3)
    else:
        return None

    # 按 ^SAMPLE 分块解析
    blocks = re.split(r"\n\^SAMPLE\s*=", txt)
    rows = []
    for b in blocks[1:]:
        def get(field):
            m = re.search(rf"^!{field}\s*=\s*(.+)$", b, re.M)
            return m.group(1).strip() if m else ""
        gsm = re.search(r"^\s*(\S+)", b)
        title = get("Sample_title")
        source = get("Sample_source_name_ch1")
        chars = " | ".join(re.findall(r"^!Sample_characteristics_ch1\s*=\s*(.+)$", b, re.M))
        info = f"{title} {source} {chars}"
        rows.append({"gsm": gsm.group(1) if gsm else "",
                     "title": title, "source": source,
                     "chars": chars[:220], "group_raw": title,
                     "group": classify(info)})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    summary = []
    for gse in GSES:
        df = fetch(gse)
        if df is None or not len(df):
            print(f"❌ {gse} 获取失败")
            continue
        vc = df["group"].value_counts().to_dict()
        print(f"\n{'='*80}\n{gse}: {len(df)} 样本 | {vc}")
        print(f"  source 样例: {df['source'].iloc[0][:80]}")
        print(f"  标题样例: {', '.join(df['title'].head(8).tolist())}")
        print(f"  characteristics 样例: {df['chars'].iloc[0][:140]}")
        summary.append({"gse": gse, "n_samples": len(df), **vc})
    if summary:
        pd.DataFrame(summary).to_csv(CACHE / "geo_group_summary.csv", index=False)
        print(f"\n✅ 汇总已保存 ({len(summary)} 个数据集)")
