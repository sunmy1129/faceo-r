from pathlib import Path

# ---------- 修正 .md ----------
p = Path('/Users/sunmy/Documents/QF/GWAS/paper/manuscript.md')
s = p.read_text(encoding='utf-8')

old = """Both null models produced modularity values *higher* than those observed on the real pathway networks (degree-preserving rewiring: mean *z* = −5.54; expression permutation: mean *z* = −5.09). That is, the observed modularity did not exceed what degree-preserving randomisation of the same graph attains. This is a substantive negative result: it implies that modularity *Q* on pathway-derived co-expression graphs should not be used as evidence that the inferred modules reflect organisation beyond what the network's degree distribution already implies. We therefore base our primary conclusions on the functional coherence metric and on downstream predictive utility, both of which are defined with respect to explicit external references (annotation databases and clinical phenotypes), rather than on modularity."""

new = """Modularity on pathway-derived graphs proved to be **unstable with respect to graph construction rather than interpretable as biological evidence**. Two diagnostics support this.

*Threshold sensitivity.* Because topological overlap matrices must be thresholded before community detection, we varied the threshold from the 50th to the 95th percentile of edge weights. The resulting modularity rose monotonically from a mean of 0.269 to 0.419 (Table S6), and the sign of the comparison against a degree-preserving null model **reversed** across this range. An earlier analysis using a 75th-percentile threshold — the configuration most commonly adopted — yielded *z* = −5.54 (observed *below* null), whereas a less sparse construction yields the opposite. Modularity comparisons against null models are therefore **threshold-dependent and cannot be reported as a robust property of the data**.

*Weight dependence.* Holding topology fixed and replacing all edge weights with a constant reduced mean modularity from 0.335 to 0.237 — a 29% contribution from weights alone (Table S6) — confirming that modularity on these graphs is substantially, though not exclusively, driven by the weight distribution rather than by topology.

Taken together, these results indicate that modularity *Q* — and, critically, its significance against null models — should not be used as primary evidence in pathway-centric module analysis, because both its magnitude and its apparent significance depend on construction choices that are rarely reported. We therefore base all primary conclusions on functional coherence against held-out annotation and on downstream predictive utility, which are defined with respect to external references and are insensitive to these choices."""

if old in s:
    s = s.replace(old, new, 1)
    # 修正 Abstract 中的对应表述
    s = s.replace(
        "observed modularity did not exceed degree-preserving null models (*z* = −5.54), so it should not be used as evidence of biological organisation",
        "modularity — and its significance against degree-preserving null models — proved unstable with respect to graph construction (threshold sensitivity reverses the sign; weights contribute 29%), so it should not be used as primary evidence", 1)
    p.write_text(s, encoding='utf-8')
    print("✅ .md §3.7 已诚实修正")
else:
    print("❌ .md 锚点未找到")
