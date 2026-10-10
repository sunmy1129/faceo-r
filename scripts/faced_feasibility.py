#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决定性实验：结构指纹能否预测最优 γ？
方法：留一交叉验证（Leave-One-Cohort-Out），特征 = 各 γ 下的结构指标（不含功能注释）
对比基线：随机猜 / 永远选最常见 γ / 永远选 γ=1.0
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
RES = "/Users/sunmy/Documents/QF/GWAS/output/core_results/full_eval_v3.csv"


def build(df):
    df = df.copy()
    df["algo"] = df["algo"].astype(str)
    lg = df[df.algo.str.startswith("Leiden γ=")].copy()
    lg["gamma"] = lg.algo.str.extract(r"γ=([0-9.]+)").astype(float)
    gs = sorted(lg.gamma.unique())
    idx = ["gse", "pathway"]
    rec = []
    for (gse, pw), sub in lg.groupby(idx):
        sub = sub.set_index("gamma").sort_index()
        if len(sub) < 6:
            continue
        row = {"gse": gse, "pathway": pw}
        # 结构指纹：各 γ 下的模块数 / 模块度 / 稳定性
        for g in gs:
            if g in sub.index:
                row[f"nmod_{g}"] = sub.loc[g, "n_modules"]
                row[f"Q_{g}"] = sub.loc[g, "modularity_Q"]
                row[f"stab_{g}"] = sub.loc[g, "stability_ARI"]
        # 通路规模
        row["n_genes"] = sub["n_genes"].iloc[0]
        row["n_samples"] = sub["n_samples"].iloc[0]
        # 目标：功能一致性最高的 γ
        row["y_gamma"] = sub["coh_z_mean"].idxmax()
        row["z_opt"] = sub["coh_z_mean"].max()
        row["z_g1"] = sub.loc[1.0, "coh_z_mean"] if 1.0 in sub.index else np.nan
        rec.append(row)
    return pd.DataFrame(rec), gs


def main():
    df = pd.read_csv(RES)
    data, gs = build(df)
    feat = [c for c in data.columns
            if c.startswith(("nmod_", "Q_", "stab_")) or c in ("n_genes", "n_samples")]
    # 去掉稳定性（可能含 NaN）
    feat = [c for c in feat if data[c].notna().all()]
    X = data[feat].values
    gs_sorted = sorted(set(data["y_gamma"]))
    y_raw = data["y_gamma"].values
    y = np.array([gs_sorted.index(v) for v in y_raw])
    cohorts = data["gse"].values
    print(f"样本: {len(data)} 组合 | 特征: {len(feat)} 维 | 类别: {len(gs_sorted)} 个 γ")

    # --- 基线 ---
    base_rand = 1.0 / len(set(y))
    from collections import Counter
    most_common = Counter(y).most_common(1)[0][0]
    acc_mode = (y == most_common).mean()
    acc_g1 = (y == 1.0).mean()
    print(f"\n基线准确率:")
    print(f"  随机猜        {base_rand:.3f}")
    print(f"  永远选最常见 γ ({most_common})  {acc_mode:.3f}")
    print(f"  永远选 γ=1.0   {acc_g1:.3f}")

    # --- 留一队列交叉验证（真正的泛化测试）---
    for name, clf_fn in [("RandomForest", lambda: RandomForestClassifier(
                            n_estimators=300, max_depth=4, random_state=42)),
                         ("LogisticReg", lambda: LogisticRegression(
                            max_iter=2000, C=0.5))]:
        preds, true, z_oracle, z_pred, z_g1 = [], [], [], [], []
        for gse in np.unique(cohorts):
            tr, te = cohorts != gse, cohorts == gse
            if tr.sum() < 20 or te.sum() < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = clf_fn()
            clf.fit(sc.transform(X[tr]), y[tr])
            p = clf.predict(sc.transform(X[te]))
            p_gamma = [gs_sorted[pi] for pi in p]
            preds.extend(p_gamma)
            true.extend([gs_sorted[yi] for yi in y[te]])
            # 增益：预测 γ 的 z  vs  γ=1.0 的 z
            sub = data[te]
            for (_, r), pp in zip(sub.iterrows(), p_gamma):
                row = df[(df.gse == r["gse"]) & (df.pathway == r["pathway"])]
                lg = row[row.algo == f"Leiden γ={pp}"]
                g1 = row[row.algo == "Leiden γ=1.0"]
                if len(lg) and len(g1):
                    z_pred.append(lg["coh_z_mean"].iloc[0])
                    z_g1.append(g1["coh_z_mean"].iloc[0])
                    z_oracle.append(r["z_opt"])
        acc = float(np.mean(np.array(true) == np.array(preds)))
        gain_pred = np.mean(np.array(z_pred) - np.array(z_g1))
        gain_oracle = np.mean(np.array(z_oracle) - np.array(z_g1))
        print(f"\n{name}（留一队列交叉验证）:")
        print(f"  预测准确率        {acc:.3f}  (vs 随机 {base_rand:.3f})")
        print(f"  达到 oracle 的     {acc/max(base_rand,1e-9):.1f}x 随机水平")
        print(f"  FACED 实际增益    +{gain_pred:.3f}  (oracle 上限 +{gain_oracle:.3f})")
        print(f"  增益回收率        {gain_pred/max(gain_oracle,1e-9)*100:.0f}%")


if __name__ == "__main__":
    main()
