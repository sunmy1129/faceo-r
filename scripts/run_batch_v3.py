#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批处理 wrapper：串行执行每个数据集，带超时强制终止
避免僵尸进程堆积占用内存
"""
import subprocess
import sys
import time
from pathlib import Path

PY = "/Users/sunmy/.hermes/venv_gwas/bin/python"
SCRIPT = "/Users/sunmy/scripts/gwas_full_analysis_v3.py"
SCRIPT = str(Path.home() / ".hermes/scripts/gwas_full_analysis_v3.py")

DATASETS = ["GSE10072", "GSE137342", "GSE15653", "GSE17536", "GSE19188",
            "GSE25066", "GSE33000", "GSE36961", "GSE39582", "GSE42568",
            "GSE61635", "GSE6764"]
import sys as _sys
if len(_sys.argv) > 1:
    DATASETS = _sys.argv[1:]

TIMEOUT = 900   # 每个数据集最多 15 分钟


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    done, failed = [], []
    for gse in DATASETS:
        log(f"▶ {gse} 开始")
        t0 = time.time()
        try:
            r = subprocess.run([PY, SCRIPT, gse, "4"],
                               timeout=TIMEOUT, capture_output=True, text=True)
            dt = time.time() - t0
            tail = [l for l in r.stdout.split("\n") if "已保存" in l or "累计" in l]
            if r.returncode == 0:
                log(f"  ✅ {gse} 完成 ({dt:.0f}s)")
                done.append(gse)
            else:
                log(f"  ⚠ {gse} 退出码 {r.returncode} ({dt:.0f}s)")
                if r.stderr:
                    log(f"     err: {r.stderr.strip().split(chr(10))[-1][:100]}")
                failed.append(gse)
            for l in tail[-2:]:
                log(f"     {l.strip()}")
        except subprocess.TimeoutExpired:
            log(f"  ⏰ {gse} 超时 >{TIMEOUT}s，强制终止")
            failed.append(gse)
    log(f"完成: {len(done)} 成功, {len(failed)} 失败")
    if failed:
        log(f"失败: {', '.join(failed)}")
    return done, failed


if __name__ == "__main__":
    main()
