#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""manuscript.md -> PDF（LaTeX 风格排版）"""
import re
import sys
from pathlib import Path
sys.path.insert(0, "/Users/sunmy/.hermes/scripts")
from gwas_report import Doc, REG, BOLD, BLACK, MID, DARK

MD = Path("/Users/sunmy/Documents/QF/GWAS/paper/manuscript.md")
PDF_OUT = Path("/Users/sunmy/Documents/QF/GWAS/paper/manuscript.pdf")


def clean(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = re.sub(r"`(.+?)`", r"\1", t)
    return t.strip()


def main():
    lines = MD.read_text(encoding="utf-8").split("\n")
    pdf = Doc(format="A4")
    pdf.add_font("r", "", REG)
    pdf.add_font("b", "", BOLD)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(22, 18, 22)
    pdf.add_page()

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("# "):
            pdf.set_font("b", size=13)
            pdf.set_text_color(*BLACK)
            pdf.ln(1)
            pdf.multi_cell(0, 6.4, clean(line[2:]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.set_font("b", size=11)
            pdf.set_text_color(*BLACK)
            pdf.ln(2.5)
            pdf.multi_cell(0, 6, clean(line[3:]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.set_font("b", size=9.6)
            pdf.set_text_color(*BLACK)
            pdf.ln(1.6)
            pdf.multi_cell(0, 5.2, clean(line[4:]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.6)
        elif line.startswith("[[FIG:"):
            # 格式：[[FIG:文件名|图注]]
            m = re.match(r"\[\[FIG:(.+?)\|(.+?)\]\]", line)
            if m:
                fname, cap = m.group(1), m.group(2)
                # 在两个图目录中查找
                cands = [Path("/Users/sunmy/Documents/QF/GWAS/output/figures_v3") / fname,
                         Path("/Users/sunmy/Documents/QF/GWAS/output/figures") / fname]
                img = next((c for c in cands if c.exists()), None)
                if img:
                    if pdf.get_y() > pdf.page_break_trigger - 80:
                        pdf.add_page()
                    w = pdf.w - pdf.l_margin - pdf.r_margin
                    pdf.image(str(img), x=pdf.l_margin, w=w)
                    pdf.ln(1)
                    pdf.set_font("r", size=7.4)
                    pdf.set_text_color(*MID)
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 4.1, clean(cap), new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(3)
        elif line.startswith("---"):
            pdf.ln(1)
        elif re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            pdf.set_font("r", size=8.6)
            pdf.set_text_color(*DARK)
            txt = re.sub(r"^\s*[-*]\s+", "", line)
            txt = re.sub(r"^\s*(\d+)\.\s+", r"\1. ", txt)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, 4.7, clean(txt), new_x="LMARGIN", new_y="NEXT")
        elif line.startswith("**") and line.endswith("**"):
            pdf.set_font("b", size=8.8)
            pdf.set_text_color(*BLACK)
            pdf.multi_cell(0, 4.8, clean(line), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("r", size=8.6)
            pdf.set_text_color(*DARK)
            pdf.multi_cell(0, 4.7, clean(line), new_x="LMARGIN", new_y="NEXT")

    pdf.output(PDF_OUT)
    print(f"已生成: {PDF_OUT}")
    print(f"页数: {pdf.page_no()}  大小: {PDF_OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
