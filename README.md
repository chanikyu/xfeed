# xFeed

**Predicting microbial cross-feeding flux from shotgun species abundance.**

<p align="center">
  <img src="examples/pipeline.png"
       alt="xFeed pipeline: species abundance → FluxMLP → cross-feeding flux, trained with 3-way concordance-aware loss"
       width="820">
</p>

<p align="center"><em>
<strong>xFeed pipeline</strong> — species abundance (2,055 species) is fed
into a compact FluxMLP (497 K parameters) that predicts cross-feeding flux
for 1,780 KEGG compounds per sample. Training uses a 3-way
concordance-aware loss: MSE for prediction accuracy, concordance hinge
loss (β=5.0) for disease–metabolite direction guided by 40 publications,
and metabolomics correlation loss (γ=0.5) for agreement with HMP2
measured metabolite levels.
</em></p>

<p align="center">
  <img src="examples/sankey.png"
       alt="xFeed Sankey — producer species → cross-feedable compounds → consumer species"
       width="820">
</p>

<p align="center"><em>
<strong>Example output</strong> — each ribbon is one predicted
cross-feeding event: a producer species (left) hands a metabolite
(centre) to a consumer species (right). Ribbon thickness is proportional
to predicted flux and colour identifies the compound. This is exactly
the figure that <code>xfeed predict</code> writes to
<code>images/sankey.png</code> in a single CLI call.
</em></p>

xFeed is a neural network that predicts per-sample, per-compound
**cross-feeding flux** for **1,780 KEGG cross-feedable compounds** directly
from a standard shotgun species abundance table. The model learns the
species → function → flux mapping from 11,688 training samples and does
not require KEGG module or capability annotations at inference time —
just abundance in, flux out.

## Key features

- **Flux-level output**: 1,780 KEGG cross-feedable compounds covering
  central carbon metabolism, SCFAs, amino acids, B-vitamins, aromatics,
  and more (derived from 4,498 KEGG REACTIONs over 951 annotated species).
- **Abundance-only input**: no KEGG profiles needed at inference time.
  The model implicitly learns the species → function mapping from
  training — it has never seen a KEGG annotation on its input side.
- **Fast**: 22,588 samples in 1.3 seconds on a laptop CPU.
- **Biologically accurate**: literature concordance 20/22 (91%) across
  13 diseases and 40 publications. HMP2 metabolomics Spearman ρ = 0.40
  (mean across 7 metabolites), butyrate ρ = 0.51, propionate ρ = 0.45.
- **Batteries included**: `xfeed predict` produces a long-format TSV, a
  dense NumPy archive, nine publication-ready figures, per-figure source
  TSVs (so you can re-plot with any tool), and one flow diagram per
  top-K species.

## Installation

### conda (recommended)

```bash
conda install -c chanikyu xfeed
```

### pip

```bash
git clone https://github.com/chanikyu/xfeed.git
cd xfeed
pip install -e .
```

**Requirements**: Python ≥ 3.11, PyTorch ≥ 2.1, numpy, pandas, requests,
tqdm, matplotlib, seaborn, networkx

## Quick start

### Step 1 — Download the profiles + pretrained model (one time)

```bash
xfeed setup
```

Downloads the pre-built cross-feedable compound list, species capability
profiles, and a pretrained FluxMLP checkpoint to `~/.xfeed/`.

### Step 2 — Predict cross-feeding flux

```bash
xfeed predict --abundance your_abundance.tsv
```

Outputs into the same directory as the input:

- `xfeed_predictions.tsv` — long-format table of (sample × compound × flux)
- `xfeed_predictions.npz` — dense (n_samples × 1,780) flux matrix for
  downstream analysis
- `images/` — nine publication-ready figures (see below)
- `data/` — the exact numbers that drive each figure, one TSV per
  figure, so you can re-plot with any tool without re-running inference

### (Optional) Retrain on your own data

```bash
xfeed train --abundance your_abundance.tsv --output-dir xfeed_model/
xfeed predict --abundance new_samples.tsv --checkpoint xfeed_model/xfeed_model.pt
```

## Input format

Species abundance table (TSV or CSV):

|  | Escherichia coli | Bacteroides fragilis | Roseburia intestinalis | ... |
|--|-----------------|---------------------|------------------------|-----|
| sample_1 | 12.5 | 8.3 | 3.1 | ... |
| sample_2 | 0.0 | 15.7 | 6.2 | ... |

- **Rows** = samples, **Columns** = species names
- Values = relative abundance (any scale; log1p is applied internally)
- First column = sample IDs (used as index)
- Supported formats: `.tsv` / `.txt` (tab-separated), `.csv` (comma-separated)

> **Important**: column headers must be **species names**
> (e.g., `Escherichia coli`), not lineage strings
> (e.g., `d__Bacteria;p__Proteobacteria;...;s__Escherichia coli`).
> If your abundance table uses GTDB lineage format, extract the species
> name from the `s__` field before running xFeed.

## Prediction outputs

### `xfeed_predictions.tsv` (long format)

| sample_id | compound | compound_name | flux_log | flux_raw |
|-----------|----------|---------------|---------:|---------:|
| sample_1 | C00033 | Acetate       | 7.12 | 1231.45 |
| sample_1 | C00246 | Butyrate      | 4.58 | 96.62 |
| sample_1 | C00163 | Propionate    | 5.21 | 181.87 |

- `compound`: KEGG Compound ID
- `compound_name`: human-readable name (fetched from KEGG, disk-cached)
- `flux_log`: predicted ln(1 + flow count)
- `flux_raw`: `expm1(flux_log)` back in raw count space

Rows below `--min-flux` (default 1.0) are filtered out so the TSV stays
sparse even though the dense matrix has 1,780 columns.

### `xfeed_predictions.npz` (dense)

NumPy archive with fields:

- `sample_ids` — (n_samples,)
- `compound_list` — (1780,) KEGG compound IDs
- `compound_names` — (1780,) matching human-readable names
- `flux_log` — (n_samples, 1780) float32
- `flux_raw` — (n_samples, 1780) float32

### `images/` — nine visualisation figures

All example images below were generated by running
`xfeed predict --abundance tests/predict_test/abundance.csv` on a
91-sample **human skin microbiome** test cohort from
**Lee et al. (2025), *Microorganisms* 13, 2491**
(DOI: [10.3390/microorganisms13112491](https://doi.org/10.3390/microorganisms13112491)).

---

#### 1. `sankey` — producer → compound → consumer flow

<p align="center">
  <img src="examples/sankey.png" alt="sankey" width="720">
</p>

#### 2. `compound_composition` — what each sample is doing

<p align="center">
  <img src="examples/compound_composition.png" alt="compound composition" width="720">
</p>

#### 3. `compound_distribution` — most exchanged metabolites

<p align="center">
  <img src="examples/compound_distribution.png" alt="compound distribution" width="720">
</p>

#### 4. `crossfeeding_degree` — who is the most central

<p align="center">
  <img src="examples/crossfeeding_degree.png" alt="crossfeeding degree" width="720">
</p>

#### 5. `heatmap` — species × species interaction density

<p align="center">
  <img src="examples/heatmap.png" alt="heatmap" width="640">
</p>

### `data/` — per-figure source TSVs

Every figure has a matching TSV under `data/`, so you can re-plot the
same numbers in any tool without re-running inference.

## Commands

### `xfeed setup`

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `~/.xfeed` | Where to save profiles and model |

### `xfeed predict`

| Argument | Default | Description |
|----------|---------|-------------|
| `--abundance` | required | Species abundance table (TSV / CSV) |
| `--checkpoint` | `~/.xfeed/xfeed_model.pt` | Path to the FluxMLP checkpoint |
| `--profile-dir` | `~/.xfeed` | Profile directory from `xfeed setup` |
| `--output` | `xfeed_predictions.tsv` | Long-format TSV output |
| `--min-flux` | `1.0` | Minimum raw flux to include in the long-format TSV |
| `--top-n` | `20` | Number of top compounds in visualizations |
| `--no-visualize` | off | Skip figure generation |
| `--image-format` | `png` | `png` or `pdf` |
| `--flux-top-species` | `10` | Number of per-species flux diagrams |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |

### `xfeed train`

| Argument | Default | Description |
|----------|---------|-------------|
| `--abundance` | required | Species abundance table (TSV / CSV) |
| `--profile-dir` | `~/.xfeed` | Profile directory from `xfeed setup` |
| `--output-dir` | `xfeed_model` | Checkpoint and training history output |
| `--epochs` | 40 | Maximum training epochs |
| `--lr` | 1e-3 | AdamW learning rate |
| `--batch-size` | 128 | Mini-batch size |
| `--patience` | 10 | Early-stop patience |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |

## Architecture

- **Input**: ln(1+x) species abundance, `(2,055,)`
- **Encoder**: `LayerNorm → Linear(2055, 128) → GELU → Dropout(0.3)`
- **Head**: `Linear(128, 1780) → Softplus`
- **Parameters**: 496,898

### 3-Way Concordance-Aware Loss

```
L = L_MSE + β · L_conc + γ · L_corr
```

| Component | Weight | Description |
|-----------|--------|-------------|
| L_MSE | 1.0 | MSE(predicted, KEGG rule-based labels) |
| L_conc | β=5.0 | Concordance hinge loss — 13 disease–metabolite direction pairs as soft prior. The model learns freely; only direction is guided. |
| L_corr | γ=0.5 | 1 − Pearson(predicted, HMP2 metabolomics) — soft regulariser for 468 samples × 7 metabolites |

## How it works

1. **KEGG REACTION backbone** — 4,498 reactions × 951 species parsed
   into substrate/product compound sets.
2. **Cross-feedable compounds** — 1,780 compounds produced by ≥1 species
   and consumed by ≥1 other, excluding 37 ubiquitous cofactors.
3. **Rule-based labels** — `flux(s,c) = |P(s,c)| × |C(s,c)| − |P(s,c) ∩ C(s,c)|`
   where P = producers active in sample s, C = consumers.
4. **FluxMLP** — learns the 1,780-dim flux vector from 2,055-dim abundance
   without KEGG features on the input side.

## Performance

Trained on 8,186 samples from curatedMetagenomicData (93 studies,
subject-level split) and evaluated on 1,700 held-out test samples:

| Metric | Value |
|--------|------:|
| Mean per-compound Pearson r | **0.506** |
| Median per-compound Pearson r | **0.638** |
| Literature concordance | **20/22 (91%)** — 13 diseases, 40 publications |
| Independent concordance (not in training) | **9/11 (82%)** |
| HMP2 butyrate Spearman ρ | **0.513** |
| HMP2 propionate Spearman ρ | **0.448** |
| HMP2 mean ρ (7 metabolites) | **0.397** |
| Compounds with Pearson > 0.5 | 884 / 1,373 (64%) |
| Robustness (5-seed) | 0.518 ± 0.006 |
| Inference time (22,588 samples) | **1.3 seconds** |

### vs MICOM/AGORA2

| | xFeed | MICOM |
|--|------:|------:|
| HMP2 butyrate ρ | **0.51** | 0.36 |
| HMP2 propionate ρ | **0.45** | 0.19 |
| 400-sample analysis time | **0.07 sec** | ~3.9 hours (est.) |
| Species coverage | 2,055 | 84 |
| Compound coverage | 1,780 | ~50 |

## Citation

If you use xFeed in your research, please cite:

> xFeed: predicting microbial cross-feeding flux from shotgun species abundance

### Example data attribution

The 91-sample abundance table shipped as
`tests/predict_test/abundance.csv` — and all example figures generated
from it in `examples/` — come from a published human skin microbiome
case study. If you reuse those files, please also cite:

> Lee, K.-C.; Lee, H.; Kim, O.-S.; Sul, W.J.; Lee, H.; Kim, H.-J.
> *Case Study on Shifts in Human Skin Microbiome During Antarctica
> Expeditions.* **Microorganisms** 2025, **13**, 2491.
> https://doi.org/10.3390/microorganisms13112491

## License

GPL-3.0 (academic / non-commercial) | commercial license available on request.
