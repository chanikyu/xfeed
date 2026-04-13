# xFeed

**Predicting microbial cross-feeding flux from shotgun species abundance.**

<p align="center">
  <img src="examples/pipeline.png"
       alt="xFeed pipeline"
       width="820">
</p>

<p align="center"><em>
Species abundance (2,055 species) → FluxMLP (497K params) → cross-feeding
flux for 1,780 KEGG compounds per sample. Trained with a 3-way
concordance-aware loss combining prediction accuracy, literature-guided
disease direction, and metabolomics correlation.
</em></p>

xFeed predicts **per-sample, per-compound cross-feeding flux** for
**1,780 KEGG compounds** directly from a species abundance table.
No KEGG annotations needed at inference — just abundance in, flux out.

## Key features

- **1,780 compounds** — SCFAs, amino acids, B-vitamins, aromatics, and more
- **Abundance-only input** — no KEGG profiles needed at inference time
- **Fast** — 22,588 samples in 1.3 seconds on a laptop CPU
- **Biologically validated** — 91% literature concordance across 13 diseases
- **Batteries included** — publication-ready figures + source TSVs in one command

## Installation

```bash
# conda (recommended)
conda install -c chanikyu xfeed

# or pip
git clone https://github.com/chanikyu/xfeed.git
cd xfeed && pip install -e .
```

**Requirements**: Python ≥ 3.11, PyTorch ≥ 2.1, numpy, pandas, matplotlib, seaborn

## Quick start

```bash
xfeed setup                                  # download model + profiles (one time)
xfeed predict --abundance your_abundance.tsv  # predict cross-feeding flux
```

### Input

Species abundance table (TSV or CSV):

|  | Escherichia coli | Bacteroides fragilis | Roseburia intestinalis | ... |
|--|-----------------|---------------------|------------------------|-----|
| sample_1 | 12.5 | 8.3 | 3.1 | ... |
| sample_2 | 0.0 | 15.7 | 6.2 | ... |

Rows = samples, columns = species names, values = relative abundance.

### Output

- `xfeed_predictions.tsv` — long-format (sample × compound × flux)
- `xfeed_predictions.npz` — dense matrix for downstream analysis
- `images/` — publication-ready figures (see below)
- `data/` — per-figure source TSVs

## Output figures — how to read them

All examples below were generated from a 91-sample human skin microbiome
cohort (Lee et al., *Microorganisms* 2025,
[doi](https://doi.org/10.3390/microorganisms13112491)).

---

### `sankey.png` — Cross-feeding flow diagram

<p align="center">
  <img src="examples/sankey.png" alt="sankey" width="720">
</p>

Three columns: **producer species** (left) → **compound** (centre) → **consumer species** (right).

- **Ribbon thickness** = predicted flux × species abundance. Thicker = more active exchange.
- **Ribbon colour** = compound identity.
- Look for **thick ribbons** to find dominant metabolic exchanges.
- Species on **both sides** for the same compound = produces and consumes it.

---

### `compound_composition.png` — Per-sample metabolic profile

<p align="center">
  <img src="examples/compound_composition.png" alt="compound composition" width="720">
</p>

Stacked bar chart — each column is one sample, each colour is a compound.

- **Similar stacks** = similar metabolic composition.
- A colour band that **shrinks or grows** between samples = compound-specific shift worth investigating.

---

### `compound_distribution.png` — Most exchanged metabolites

<p align="center">
  <img src="examples/compound_distribution.png" alt="compound distribution" width="720">
</p>

Compounds ranked by producer–consumer pair count.

- **Top** = high bandwidth, robust to perturbation (many species can substitute).
- **Bottom** = narrow pathways that break if one species is lost.

---

### `crossfeeding_degree.png` — Keystone species

<p align="center">
  <img src="examples/crossfeeding_degree.png" alt="crossfeeding degree" width="720">
</p>

Species ranked by total cross-feeding degree.

- **Top bars** = keystone candidates — removing them would disrupt the network most.
- Bar colour = each species' top compound.

---

### `heatmap.png` — Species interaction density

<p align="center">
  <img src="examples/heatmap.png" alt="heatmap" width="640">
</p>

Species × species matrix — how many compounds each pair can exchange.

- **Dark red** = strong mutualistic candidates (dozens of exchangeable compounds).
- **Yellow** = metabolically isolated species.

---

## Performance

| Metric | Value |
|--------|------:|
| Literature concordance | **20/22 (91%)** — 13 diseases, 40 publications |
| Independent validation | **9/11 (82%)** — pairs not used in training |
| HMP2 butyrate Spearman ρ | **0.51** |
| HMP2 mean ρ (7 metabolites) | **0.40** |
| Mean Pearson r (1,373 compounds) | **0.506** |
| Robustness (5-seed) | 0.518 ± 0.006 |
| Inference (22,588 samples) | **1.3 sec** |

## Commands

| Command | Description |
|---------|-------------|
| `xfeed setup` | Download model + profiles to `~/.xfeed/` |
| `xfeed predict --abundance FILE` | Predict flux + generate figures |
| `xfeed train --abundance FILE` | Retrain on your own data |

Key options for `predict`: `--top-n` (compounds in figures, default 20),
`--image-format` (`png`/`pdf`), `--no-visualize` (skip figures),
`--device` (`auto`/`cpu`/`cuda`/`mps`).

## Citation

> xFeed: predicting microbial cross-feeding flux from shotgun species abundance

Example data: Lee et al. (2025), *Microorganisms* 13, 2491.

## License

GPL-3.0 (academic / non-commercial) | commercial license available on request.
