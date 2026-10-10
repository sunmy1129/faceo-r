#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合成基准 v2（提高区分度）
基准 1  LFR 风格块模型：社区清晰度梯度 μ = 0.10–0.70（更宽、更难）
基准 2  子群嵌套结构：模块内含功能子群，检验多分辨率拆分能力（关键证据）
基准 3  噪声梯度：失效阈值
输出：synthetic_results.csv
"""
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import gwas_full_analysis as GA

warnings.filterwarnings("ignore")
OUT = Path("/Users/sunmy/Documents/QF/GWAS/output/core_results")
SEED = 42


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------- 生成器 ----------
def make_block_tom(n, k, mu, seed=SEED):
    """块模型：μ 越大社区越模糊（权重重叠度增加）"""
    rng = np.random.default_rng(seed)
    sizes = np.full(k, n // k)
    sizes[:n - sizes.sum()] += 1
    labels = np.repeat(np.arange(k), sizes)
    tom = np.zeros((n, n))
    # 让权重范围随 μ 重叠：μ 越大，社区内外权重越接近
    lo_in, hi_in = 0.35 + 0.35 * mu, 1.0
    lo_out, hi_out = 0.02, 0.15 + 0.75 * mu
    for i in range(n):
        for j in range(i + 1, n):
            same = labels[i] == labels[j]
            # 边存在概率固定 0.5，靠权重区分（更难）
            if rng.random() < 0.5:
                lo, hi = (lo_in, hi_in) if same else (lo_out, hi_out)
                v = lo + (hi - lo) * rng.random()
                tom[i, j] = tom[j, i] = v
    return tom, labels


def make_nested_tom(n_mod=6, mod_size=20, n_sub=2, w_within=0.80, w_between=0.12, seed=SEED):
    """
    嵌套结构：每个模块含 n_sub 个功能子群
      子群内权重高（w_within），子群间权重低（w_between），模块间近零
    返回 TOM, 子群标签（细）, 模块标签（粗）
    """
    rng = np.random.default_rng(seed)
    n = n_mod * mod_size
    sub_per = mod_size // n_sub
    mod_lab, sub_lab = [], []
    for m in range(n_mod):
        for s in range(n_sub):
            mod_lab += [m] * sub_per
            sub_lab += [m * n_sub + s] * sub_per
    mod_lab, sub_lab = np.array(mod_lab), np.array(sub_lab)
    tom = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if mod_lab[i] != mod_lab[j]:
                p, lo, hi = 0.03, 0.01, 0.08
            elif sub_lab[i] == sub_lab[j]:
                p, lo, hi = 0.85, w_within * 0.9, 1.0
            else:
                p, lo, hi = 0.35, w_between * 0.5, w_between * 1.6
            if rng.random() < p:
                v = lo + (hi - lo) * rng.random()
                tom[i, j] = tom[j, i] = v
    return tom, sub_lab, mod_lab


def evalu(tom, true_labels, tag="", extra=None, algos=None):
    g = GA.to_ig(tom, [f"g{i}" for i in range(tom.shape[0])])
    rows = []
    for aname, afn in (algos or GA.ALGOS).items():
        try:
            lab = np.asarray(afn(g))
        except Exception:
            continue
        n_mod = len(set(lab[lab >= 0]))
        row = {"benchmark": tag, "algo": aname.replace("\n", " "),
               "ARI": adjusted_rand_score(true_labels, lab),
               "NMI": normalized_mutual_info_score(true_labels, lab),
               "n_modules": n_mod,
               "n_true": len(set(true_labels))}
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def bench_lfr(n=140, k=7, mus=(0.10, 0.25, 0.40, 0.55, 0.70), seeds=(42, 7, 123)):
    rows = []
    for mu in mus:
        for sd in seeds:
            tom, true = make_block_tom(n, k, mu, seed=sd)
            rows += evalu(tom, true, "LFR", {"mu": mu, "seed": sd, "n_true": k})
        log(f"  LFR μ={mu}")
    return rows


def bench_nested(seeds=(42, 7, 123)):
    rows = []
    for sd in seeds:
        tom, sub, mod = make_nested_tom(seed=sd)
        rows += evalu(tom, sub, "Nested_sub", {"seed": sd, "n_true": len(set(sub))})
        rows += evalu(tom, mod, "Nested_mod", {"seed": sd, "n_true": len(set(mod))})
        log(f"  Nested seed={sd}")
    return rows


def bench_noise(n=140, k=7, levels=(0.0, 0.10, 0.20, 0.35, 0.50), seeds=(42, 7, 123)):
    rows = []
    for lv in levels:
        for sd in seeds:
            tom, true = make_block_tom(n, k, 0.25, seed=sd)
            if lv > 0:
                rng = np.random.default_rng(sd)
                tom = tom * (1 + rng.normal(0, lv, tom.shape))
                tom = np.clip((tom + tom.T) / 2, 0, None)
                np.fill_diagonal(tom, 0)
            rows += evalu(tom, true, "Noise", {"noise": lv, "seed": sd, "n_true": k})
        log(f"  Noise σ={lv}")
    return rows


def main():
    t0 = time.time()
    rows = []
    log("基准 1：LFR 清晰度梯度")
    rows += bench_lfr()
    log("基准 2：嵌套子群结构")
    rows += bench_nested()
    log("基准 3：噪声梯度")
    rows += bench_noise()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic_results.csv", index=False)
    log(f"✅ 已保存 ({len(df)} 行, {time.time()-t0:.0f}s)")

    pd.set_option("display.width", 250)
    print("\n" + "="*108)
    print("基准 1  LFR：绝对准确率（ARI）随社区清晰度变化")
    print("="*108)
    print(df[df.benchmark == "LFR"].groupby(["algo", "mu"])["ARI"].mean().unstack().round(3))
    print("\n" + "="*108)
    print("基准 2  嵌套子群：能否恢复「细」子群（ARI vs 子群标签）")
    print("="*108)
    nsub = df[df.benchmark == "Nested_sub"].groupby("algo").agg(
        ARI_sub=("ARI", "mean"), 模块数=("n_modules", "mean")).round(3)
    nmod = df[df.benchmark == "Nested_mod"].groupby("algo")["ARI"].mean().round(3)
    j = nsub.join(nmod.rename("ARI_mod")).sort_values("ARI_sub", ascending=False)
    print(j)
    print("\n" + "="*108)
    print("基准 3  噪声梯度：ARI 随噪声水平")
    print("="*108)
    print(df[df.benchmark == "Noise"].groupby(["algo", "noise"])["ARI"].mean().unstack().round(3))
    return df


if __name__ == "__main__":
    main()
