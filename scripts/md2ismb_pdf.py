#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""manuscript.md -> ISMB 双栏投稿 PDF"""
import re
import sys
from pathlib import Path
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
from ismb_twocol import TwoCol, REG, BOLD

MD = Path("/Users/sunmy/Documents/QF/GWAS/paper/manuscript.md")
OUT = Path("/Users/sunmy/Documents/QF/GWAS/paper/manuscript_ismb_preview.pdf")
FIGD = [Path("/Users/sunmy/Documents/QF/GWAS/output/figures_v3"),
        Path("/Users/sunmy/Documents/QF/GWAS/output/figures")]


def clean(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = re.sub(r"`(.+?)`", r"\1", t)
    t = re.sub(r"^\s*[-*]\s+", "\u2022  ", t)
    return t.strip()


def find_img(name):
    for d in FIGD:
        p = d / name
        if p.exists():
            return str(p)
    return None


def main():
    text = MD.read_text(encoding="utf-8")
    lines = text.split("\n")
    pdf = TwoCol()
    pdf.add_font("r", "", REG)
    pdf.add_font("b", "", BOLD)
    pdf.add_page()

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("# "):
            pdf.set_font("b", size=13.5); pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 7, clean(ln[2:]), align="C",
                           new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2); pdf.y_col = pdf.get_y(); pdf.col = 0
        elif ln.startswith("## "):
            pdf.heading(clean(ln[3:]), size=9.8, space_before=3)
        elif ln.startswith("### "):
            pdf.heading(clean(ln[4:]), size=8.8, space_before=2.4)
        elif ln.startswith("[[FIG:"):
            m = re.match(r"\[\[FIG:(.+?)\|(.+?)\]\]", ln)
            if m:
                pdf.figure(find_img(m.group(1)), clean(m.group(2)))
        elif ln.startswith("---"):
            pdf.space(1.5)
        elif ln.startswith("**") and ln.rstrip().endswith("**"):
            pdf.para(clean(ln), size=8.0, bold=True, lh=4.3)
        else:
            pdf.para(clean(ln), size=8.0, lh=4.3)
        i += 1

    pdf.output(OUT)
    print(f"已生成: {OUT}")
    print(f"页数: {pdf.page_no()}  大小: {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
