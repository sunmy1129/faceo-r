from pathlib import Path

p = Path('/Users/sunmy/Documents/QF/GWAS/paper/manuscript_ismb.tex')
s = p.read_text(encoding='utf-8')
n = 0

# 找 §3.7 的正文（从 subsection 标题到 figure 环境之前）
import re
start = s.find(r"\subsection{Null models constrain the interpretation of modularity}")
if start < 0:
    print("❌ 未找到 §3.7 标题")
    raise SystemExit
# 找下一个 \begin{figure} 作为结束点
fstart = s.find(r"\begin{figure}", start)
end_of_para = s.rfind("\n\n", start, fstart)

new_body = r"""\noindent\textbf{Threshold sensitivity.} Because topological overlap matrices must be
thresholded before community detection, we varied the threshold from the 50th to the 95th
percentile of edge weights. Mean modularity rose monotonically from $0.269$ to $0.419$
(Table~S6), and the sign of the comparison against a degree-preserving null model
\emph{reversed} across this range: an earlier configuration using the 75th percentile --- the
setting most commonly adopted --- gave $z=-5.54$ (observed \emph{below} null), whereas a less
sparse construction gives the opposite. Modularity comparisons against null models are
therefore threshold-dependent and \emph{cannot} be reported as a robust property of the data.

\noindent\textbf{Weight dependence.} Holding topology fixed and replacing all edge weights
with a constant reduced mean modularity from $0.335$ to $0.237$ --- a 29\% contribution from
weights alone (Table~S6) --- confirming that modularity on these graphs is substantially,
though not exclusively, driven by the weight distribution rather than by topology.

Taken together, modularity $Q$ --- and, critically, its significance against null models ---
should not be used as primary evidence in pathway-centric module analysis, because both its
magnitude and its apparent significance depend on construction choices that are rarely
reported. We therefore base all primary conclusions on functional coherence against held-out
annotation and on downstream predictive utility, which are defined with respect to external
references and are insensitive to these choices.
"""

s = s[:start] + r"\subsection{Modularity is unstable with respect to graph construction}" + s[start + len(r"\subsection{Null models constrain the interpretation of modularity}"):end_of_para] + "\n\n" + new_body + s[end_of_para:]
n += 1

# 更新 Abstract
old_a = r"""observed modularity did not exceed degree-preserving null models ($z = -5.54$), so it should
not be used as evidence of biological organisation."""
new_a = r"""modularity --- and its significance against degree-preserving null models --- proved unstable
with respect to graph construction (threshold sensitivity reverses the sign; weights
contribute 29\%), so it should not be used as primary evidence."""
if old_a in s:
    s = s.replace(old_a, new_a, 1); n += 1

p.write_text(s, encoding='utf-8')
print(f'完成 {n} 处更新')
