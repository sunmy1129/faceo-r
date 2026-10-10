#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生物学验证：FACEO-R 模块的 GO 富集"""
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
from scipy.stats import hypergeom

R = Path("/Users/sunmy/Documents/QF/GWAS/output/core_results")
GS = Path.home() / ".hermes/gwas_data"
O = Path("/Users/sunmy/Documents/QF/GWAS/output/figures_v3")


def enrich(mod_genes, bg, go2t, min_size=5):
    out = []
    N = len(bg)
    for term, genes in go2t.items():
        gs = genes & bg
        if len(gs) < min_size or len(gs) > 400:
            continue
        k = len(mod_genes & gs)
        if k < 2:
            continue
        p = hypergeom.sf(k - 1, N, len(gs), len(mod_genes & bg))
        out.append((term, k, len(gs), p))
    out.sort(key=lambda t: t[3])
    return out


def main():
    import gwas_full_analysis as GA
    import gwas_full_analysis_v3 as V3
    import gwas_faceo_r as FR

    go = json.load(open(GS / "GO_Biological_Process_2021.json"))
    go2t = {k: set(str(g).upper() for g in v) for k, v in go.items()}
    res = pd.read_csv(R / "full_eval_v3.csv")
    gses = sorted(res.gse.unique())[:4]
    rows = []
    for gse in gses:
        raw = pd.read_csv(V3.BASE / f"qcdata/{gse}.csv", index_col=0,
                          quoting=3, low_memory=False)
        raw.index = [str(i).strip().strip('"').upper() for i in raw.index]
        df = V3.preprocess_robust(raw)
        allg = set(df.index)
        pw = V3.select_pathways(go2t, allg)
        print(f"  {gse}: {len(allg)} 基因, {len(pw)} 通路")
        for name, genes in pw[:3]:
            genes = [g for g in genes if g in df.index]
            if not (60 <= len(genes) <= 200):
                continue
            sub = df.loc[genes]
            tom = np.corrcoef(np.nan_to_num(sub.values.astype(float)))
            np.fill_diagonal(tom, 1.0)
            tom = (tom + tom.T) / 2
            q = FR.graph_reliability(tom)
            mods, hard, info = FR.faceo_r(tom, genes, go2t, reliability=q)
            gnames = list(sub.index)
            for mi, m in enumerate(mods):
                mgs = {gnames[int(i)] for i in m if int(i) < len(gnames)}
                en = enrich(mgs, allg, go2t)
                if not en:
                    continue
                base = name.split(" (GO")[0].strip().lower()
                en2 = [t for t in en if t[0].split(" (GO")[0].strip().lower() != base]
                term, k, tsz, p = en2[0] if en2 else en[0]
                rows.append({"gse": gse, "pathway": name[:34], "module": mi,
                             "n": len(mgs), "top_GO": term[:52], "k": k,
                             "term_size": tsz, "pval": p})
    d = pd.DataFrame(rows)
    d.to_csv(R / "bio_validation.csv", index=False)
    if len(d) == 0:
        print("⚠ rows 为空 —— 检查上游")
        return
    sig = d[d["pval"] < 0.05]
    print(f"模块总数: {len(d)}  显著富集(<0.05): {len(sig)}  占 {100*len(sig)/max(len(d),1):.1f}%")
    print(f"中位 p: {d.pval.median():.2e}   最小 p: {d.pval.min():.2e}")
    print("\n示例（每队列首个模块）:")
    for _, r in d.groupby("gse").head(1).iterrows():
        print(f"  {r.gse}: {r.n}基因 -> {r.top_GO} (k={r.k}/{r.term_size}, p={r.pval:.1e})")


if __name__ == "__main__":
    main()
