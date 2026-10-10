#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISMB 投稿规格双栏 PDF 渲染器（fpdf2 实现，无需 LaTeX）"""
import re
import warnings
from pathlib import Path
from fpdf import FPDF

warnings.filterwarnings("ignore")

REG = "/Users/sunmy/.hermes/fonts/SongtiSC-Regular.ttf"
BOLD = "/Users/sunmy/.hermes/fonts/SongtiSC-Bold.ttf"
BLACK, DARK, MID = (0, 0, 0), (30, 30, 30), (110, 110, 110)


class TwoCol(FPDF):
    """A4 双栏文档：左栏填满后转右栏，再新页"""

    def __init__(self):
        super().__init__(format="A4")
        self.lm, self.rm = 16, 16
        self.tm, self.bm = 16, 16
        self.gutter = 7
        self.col_w = (self.w - self.lm - self.rm - self.gutter) / 2
        self.col = 0
        self.y_col = self.tm

    def col_x(self, c=None):
        c = self.col if c is None else c
        return self.lm + c * (self.col_w + self.gutter)

    def next_col(self):
        if self.col == 0:
            self.col = 1
            self.y_col = self.tm
        else:
            self.add_page()
            self.col = 0
            self.y_col = self.tm

    def space(self, h, needed=None):
        """在当前栏预留空间，不足则换栏"""
        if self.y_col + (needed or h) > self.h - self.bm - 14:
            self.next_col()
        self.y_col += h

    def para(self, text, size=8.0, lh=4.3, bold=False, color=DARK, indent=0,
             align="L"):
        self.set_font("b" if bold else "r", size=size)
        self.set_text_color(*color)
        w = self.col_w - indent
        # 预计算高度
        lines = self.multi_cell(w, lh, text, dry_run=True, output="LINES")
        h = len(lines) * lh
        if self.y_col + h > self.h - self.bm - 14:
            self.next_col()
        self.set_xy(self.col_x() + indent, self.y_col)
        self.multi_cell(w, lh, text, align=align, new_x="LMARGIN", new_y="NEXT")
        self.y_col = self.get_y()
        self.set_xy(self.col_x(), self.y_col)

    def heading(self, text, size=9.5, space_before=2.2):
        self.space(space_before + 5.5, needed=space_before + 12)
        self.y_col += space_before
        self.set_font("b", size=size)
        self.set_text_color(*BLACK)
        self.set_xy(self.col_x(), self.y_col)
        self.multi_cell(self.col_w, 5.2, text, new_x="LMARGIN", new_y="NEXT")
        self.y_col = self.get_y()

    def figure(self, img, caption, size=7.0):
        """跨栏大图"""
        if img is None or not Path(img).exists():
            return
        full_w = self.w - self.lm - self.rm
        # 估算图高
        try:
            from PIL import Image
            iw, ih = Image.open(img).size
            h = full_w * ih / iw
        except Exception:
            h = 60
        if h > self.h - self.tm - self.bm - 30:
            h = self.h - self.tm - self.bm - 30
        # 大图强制新页
        self.add_page()
        self.col = 0
        self.y_col = self.tm
        self.image(img, x=self.lm, y=self.y_col, w=full_w)
        self.y_col += h + 1.5
        self.set_font("r", size=size)
        self.set_text_color(*MID)
        self.set_xy(self.lm, self.y_col)
        self.multi_cell(full_w, 4.0, caption, new_x="LMARGIN", new_y="NEXT")
        self.y_col = self.get_y() + 2
