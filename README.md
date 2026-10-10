# FACEO-R: Functionally Anchored Overlapping Community Detection for Robust Pathway Module Discovery

Reproducible pipeline for the manuscript:

> **FACEO-R: Functionally Anchored Overlapping Community Detection for Robust Pathway Module
> Discovery** — *A systematic benchmark of 31 partitioning algorithms under controlled data
> degradation*

## Overview

The field relies on three implicit assumptions when partitioning the genes of a pathway into
functional modules: that a single partitioning algorithm is adequate, that a single resolution
is defensible, and that the modularity *Q* of the resulting modules evidences biological
organisation. This pipeline tests all three by benchmarking **31 partitioning algorithms** on
identical topological-overlap (TOM) graphs derived from GO biological-process pathway
subnetworks across **16 GEO expression cohorts**, evaluated with a six-layer protocol
(structure, functional coherence, stability, robustness, biological validity, downstream
prediction), plus synthetic benchmarks with ground truth.

It further implements **FACEO-R**, a functionally anchored overlapping community detection
method in which the weight of GO-derived functional similarity adapts to the estimated
reliability of the expression graph. Two properties follow by construction: non-inferiority on
clean data (it collapses to standard Leiden as graph reliability approaches unity) and
robustness under degradation.

## Environment

Python 3.11. Dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `igraph`, `leidenalg`,
`networkx`, `python-louvain`, `umap-learn`, `hdbscan`, `statsmodels`, `fpdf2`,
`torch`, `torch_geometric`, `mygene`, `gseapy`, `scanpy`, `anndata`.

## Pipeline

| Step | Script | Purpose |
|---|---|---|
| 1 | `gwas_expand_datasets3.py`, `add_new_cohorts.py` | Download GEO series matrices and map probes → gene symbols (mygene.info) |
| 2 | `fix_csv_quotes.py`, `fix_annotate.py` | Normalise CSV quoting; re-annotate probe-level matrices |
| 3 | `gwas_full_analysis.py` | Base analysis engine (17 algorithms, six evaluation layers) |
| 4 | `gwas_algorithms_v2.py` | Additional 20 algorithms (new families) |
| 5 | `gwas_selfsup.py` | Self-supervised graph embeddings (edge-prediction objective) |
| 6 | `gwas_l2_functional.py` | L2 functional-coherence layer (size-unbiased z-score) |
| 7 | `gwas_full_analysis_v3.py` | Main engine: 31 algorithms × all cohorts |
| 8 | `run_batch_v3.py` | Per-cohort runner with timeout isolation (accepts cohort args) |
| 9 | `gwas_synthetic.py` | Synthetic benchmarks with ground truth |
| 10 | **`gwas_faceo_r.py`** | **FACEO-R: functionally anchored overlapping community detection** |
| 11 | **`faceo_r_experiment.py`** | **Degradation experiments (sampling / noise / dropout)** |
| 12 | **`scrna_validation.py`** | **Real single-cell validation (GSE145926)** |
| 13 | **`weight_dependence.py`** | **Modularity threshold-sensitivity and weight-contribution analysis** |
| 14 | `gwas_downstream_v3.py` | Downstream clinical prediction (L6) |
| 15 | `gwas_merge_v3.py` | Merge results and run paired statistics |
| 16 | `plot_*_v2.py` | Journal-grade figures 1–12 |
| 17 | `gwas_supplementary.py` | Supplementary tables/figures |
| 18 | `tectonic manuscript_ismb.tex` | ISMB-format PDF (A4, two-column, zero overfull) |

## Running

```bash
# Main benchmark (per cohort, isolated processes)
python run_batch_v3.py                 # all cohorts
python run_batch_v3.py GSE38642 ...    # or specific cohorts

# Merge + statistics
python gwas_merge_v3.py

# Synthetic benchmarks / downstream
python gwas_synthetic.py
python gwas_downstream_v3.py

# FACEO-R: degradation experiments + real single-cell validation
python faceo_r_experiment.py
python scrna_validation.py

# Modularity diagnostics
python weight_dependence.py

# Figures
python plot_main_v2.py && python plot_heat_v2.py && python plot_faceo_r_v2.py
python plot_scrna_v2.py && python plot_null_v2.py

# Manuscript
cd paper && tectonic manuscript_ismb.tex
```

All scripts use a fixed random seed (42). Stochastic algorithms are additionally assessed
across bootstrap resamples rather than seeds alone.

## Results

Evaluation tables are in `results/`:

| File | Records | Content |
|---|---|---|
| `full_eval_v3.csv` | 3,100 | Main benchmark (16 cohorts × 9 pathways × 31 algorithms) |
| `synthetic_results.csv` | 528 | Synthetic benchmarks (block model, nested sub-structure, noise) |
| `downstream_v3.csv` | 317 | Downstream clinical prediction |
| `faceo_r_results.csv` | 495 | FACEO-R degradation experiments (165 comparisons × 3 methods) |
| `scrna_validation.csv` | 60 | Real single-cell validation (16 subnetworks × 3 methods) |
| `weight_dependence.csv` | 200 | Modularity threshold sensitivity (40 pairs × 5 thresholds) |
| `statistics_v3.csv` | — | Paired tests (Wilcoxon + BH-FDR + Cliff's δ) |

## Key results

**Main benchmark** (100 pathway–cohort pairs):

| Metric | Leiden γ=1.0 | WGCNA-style | p (FDR) | Cliff δ |
|---|---|---|---|---|
| Modules | 4.50 | 3.00 | 4.9e-16 | +0.870 |
| Modularity Q | 0.291 | 0.000 | 2.0e-17 | +1.000 |
| Functional coherence z | 0.689 | 0.045 | 4.6e-07 | +0.418 |
| Antagonistic separation | 0.500 | 0.000 | 7.0e-07 | +0.727 |
| Downstream AUC | 0.989 | 0.898 | — | — |
| Stability | 0.646 | 0.626 | 0.068 (n.s.) | +0.133 |

**FACEO-R** (functionally anchored, adaptive weight):

| Setting | vs Leiden γ=1.0 | vs WGCNA-style |
|---|---|---|
| Controlled degradation (165 comparisons) | 75% win, p = 9.2e-12 | 82% win, p = 4.9e-19 |
| Feature dropout 50% | 88% win, Δ = +0.046 | — |
| **Real single-cell (GSE145926)** | **69% win, p = 0.029** | **94% win, p = 6.1e-05** |

**Methodological findings:**

- Modularity is **unstable with respect to graph construction**: varying the edge-weight
  threshold from the 50th to the 95th percentile moves mean *Q* from 0.269 to 0.419 (+56%)
  and *reverses* the sign of the comparison against a degree-preserving null model. Weights
  contribute 29% of *Q* independently of topology.
- The functionally optimal resolution is **not predictable**: optimal γ is highly dispersed
  across pathways (entropy 1.74 of a maximum 1.79) and four families of automatic selection
  strategies all failed (accuracy 0.16–0.46, none significantly above chance).
- Self-supervised embeddings reproduce the Leiden partition several-fold better than the
  hierarchical alternative (ARI 0.298 versus 0.048).

## Notes and known limitations

- Four high-complexity algorithms (edge betweenness, link communities, optimal modularity,
  full multi-scale Markov Stability) are excluded because they exceed practical runtime and
  destabilise the igraph C layer at the pathway sizes considered.
- GSE33000 (Agilent platform) and GSE26927 are excluded because no probe annotation is
  retrievable; GSE89408 and GSE19804 are excluded because their series matrices contain no
  expression values.
- The single-cell validation applies the bulk protocol to real scRNA-seq (cells as samples,
  true zero fractions 51–89%); the advantage there is **not** correlated with zero fraction
  (ρ = −0.28, n.s.), so the benefit derives from orthogonal functional information rather than
  from compensating dropout per se.
- Density-based algorithms operate in a spectral embedding space to mitigate the curse of
  dimensionality on high-dimensional similarity profiles.

## Citation

If you use this code, please cite the manuscript above.

## License

To be specified on publication.
