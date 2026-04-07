# xFeed

**Predicting microbial cross-feeding flux from shotgun species abundance.**

<p align="center">
  <img src="examples/architecture.png" alt="xFeed model architecture" width="480">
</p>

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
- **Fast**: predicts thousands of samples in a few seconds on a laptop CPU.
- **Biologically accurate**: SCFAs (acetate r = 0.77, butyrate r = 0.73,
  propionate r = 0.62), amino acids (mean r = 0.78 across seven), B-vitamins
  (biotin r = 0.79), and the gut-brain indole axis (r = 0.76).
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

### `images/` — nine figures

| Figure | What it shows |
|--------|---------------|
| `variable_compounds` | Top-20 compounds ranked by **coefficient of variation across samples** — the metabolites that actually differ from sample to sample. Each row is a strip plot of the samples' predicted fluxes, with the cohort mean marked and the CV value annotated. Replaces the earlier `top_compounds` figure, which always highlighted universal metabolites. |
| `flux_pca` | 2-D PCA embedding of the `(n_samples × 1,780)` predicted log-flux matrix via centred SVD. Samples are coloured by total log-flux and the top outliers are labelled. Axes report the explained variance ratio of each principal component. Needs ≥ 3 samples. |
| `flux_density` | Histogram + rank plot of the total predicted flux per sample. |
| `compound_composition` | Stacked bar chart of the per-sample top-N compound composition. |
| `compound_categories` | Two-panel view of the per-category flux totals and distribution (SCFA, amino acid, vitamin, sugar, cofactor, aromatic, other). |
| `heatmap` | Top-20 species × top-20 species matrix of cross-feeding interaction counts (number of compounds exchanged between each pair in either direction), coloured in `YlOrRd`. |
| `compound_distribution` | Horizontal bar chart of the compounds with the most producer-consumer species pairs in the current cohort, labelled with raw interaction counts. |
| `crossfeeding_degree` | Horizontal bar chart of the species with the highest total cross-feeding degree (producer + consumer), coloured by each species' top compound. |
| `sankey` | Three-column Sankey diagram: producer species → compound → consumer species, for the top compounds by predicted flux. |

Additionally, `images/flux/` contains one **per-species flow diagram**
(`Escherichia_coli.png`, `Bifidobacterium_longum.png`, …) for the top-K
most abundant species in your cohort. Each diagram has four columns —
upstream compounds consumed, the species itself, downstream compounds
produced, and the other species that consume those downstream products —
so you can read off each species' place in the cross-feeding network at
a glance.

### `data/` — per-figure source TSVs

Every figure has a matching TSV under `data/`, so you can re-plot the
same numbers in any tool without re-running inference.

| File | Content |
|------|---------|
| `data/compound_ranking.tsv` | Master table of all 1,780 compounds: name, category, mean flux (log & raw), std, coefficient of variation. |
| `data/variable_compounds.tsv` | Top-20 CV-ranked compounds with one `log_{sample_id}` column per sample — the exact values plotted in the strip figure. |
| `data/flux_pca_scores.tsv` | Per-sample PC1–PC5 scores and total log-flux. |
| `data/flux_pca_variance.tsv` | Explained variance ratio for each of the top 5 principal components. |
| `data/flux_density.tsv` | Per-sample total flux (raw & log), mean flux, number of non-zero compounds. |
| `data/compound_composition.tsv` | Top-20 × samples long-format (sample, compound, name, flux_raw, flux_log). |
| `data/compound_categories.tsv` | Per-category aggregated totals and means. |
| `data/heatmap_interaction_counts.tsv` | Top-20 × top-20 symmetric species-pair interaction matrix. |
| `data/compound_distribution.tsv` | Per-compound producer × consumer counts across the present species pool. |
| `data/crossfeeding_degree.tsv` | Per-species total cross-feeding degree and top compound. |
| `data/sankey_edges.tsv` | Long-format (compound, producer, consumer, flux_weight) edges for the Sankey figure. |
| `data/flux/{species}.tsv` | Four-direction flow for one species: `direction ∈ {consumed, produced, downstream_consumer}`, compound, partner, weight. |

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
| `--profile-dir` | `~/.xfeed` | Profile directory from `xfeed setup` (needed for the `heatmap`, `compound_distribution`, `crossfeeding_degree`, `sankey`, and per-species flux figures) |
| `--output` | `xfeed_predictions.tsv` | Long-format TSV output (the dense `.npz`, `images/` and `data/` use the same stem) |
| `--min-flux` | `1.0` | Minimum raw flux to include in the long-format TSV |
| `--top-n` | `20` | Number of top compounds to use in the rank-based figures |
| `--no-visualize` | off | Skip figure / data TSV generation |
| `--image-format` | `png` | `png` (raster, good for web/slides) or `pdf` (vector, good for manuscripts). All figures in a run use the chosen format. |
| `--flux-top-species` | `10` | Number of most-abundant species to draw per-species flux diagrams for, in `images/flux/`. Set to `0` to skip. |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |

### `xfeed train`

| Argument | Default | Description |
|----------|---------|-------------|
| `--abundance` | required | Species abundance table (TSV / CSV) |
| `--profile-dir` | `~/.xfeed` | Profile directory from `xfeed setup` |
| `--output-dir` | `xfeed_model` | Checkpoint and training history output |
| `--epochs` | 40 | Maximum training epochs (early stop with patience) |
| `--lr` | 1e-3 | AdamW learning rate with cosine annealing |
| `--batch-size` | 128 | Mini-batch size |
| `--patience` | 10 | Early-stop patience |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |

## Architecture

<p align="center">
  <img src="examples/architecture.png" alt="xFeed FluxMLP architecture" width="480">
</p>

- **Input**: log1p species abundance, `(2,055,)`
- **Encoder**: stacked `LayerNorm → Linear → GELU → Dropout` blocks,
  `2055 → 1024 → 512 → 256 → 128`
- **Head**: `Linear(128, 1780)` → `softplus` to keep flux non-negative
- **Parameters**: ~3.0 M
- **Loss**: MSE on ln(1 + flow count)
- **No KEGG modules or capability annotations on the input side**

## How it works

1. **KEGG REACTION backbone** — every reaction catalysed by a KO in a
   species' KEGG genome annotation is parsed into substrate / product
   compound sets. 4,498 reactions × 951 species.
2. **Cross-feedable compounds** — 1,780 compounds that are produced by
   at least one species and consumed by at least one other, excluding
   37 ubiquitous cofactors (H₂O, ATP, NAD⁺, CoA, …).
3. **Rule-based labels** — for each training sample, the flux for a
   compound equals (# producer species active) × (# consumer species
   active) − (# species that do both). These are the targets FluxMLP
   learns to predict from abundance alone.
4. **FluxMLP** — a 3.0 M-parameter feed-forward network learns the
   1,780-dim flux vector from the 2,055-dim log1p species abundance
   vector, without any KEGG feature engineering on the input side.

## Performance

Trained on 8,186 samples from the curatedMetagenomicData collection
(subject-level split across 84 studies) and evaluated on a held-out
test set of 1,700 samples:

| Metric | Value |
|--------|------:|
| Mean per-compound Pearson r | **0.484** |
| Median per-compound Pearson r | **0.618** |
| Predictable compounds (non-zero test variance) | 1,373 / 1,780 |
| Compounds with Pearson > 0.5 | 832 / 1,373 (60.6 %) |
| Compounds with Pearson > 0.7 | 516 / 1,373 (37.6 %) |
| Log-MSE vs. mean baseline | **−17.3 %** |
| Mean Pearson vs. nearest-neighbour baseline | **+63.9 %** |
| Mean Spearman vs. nearest-neighbour baseline | **+79.9 %** |

Biologically important metabolites (test-set Pearson r):

| Compound | Pearson r |
|----------|----------:|
| β-Alanine | 0.81 |
| Biotin (B7) | 0.79 |
| D-Glucose | 0.78 |
| L-Glutamate | 0.77 |
| **Acetate** | **0.77** |
| L-Tryptophan | 0.77 |
| Indole | 0.76 |
| Thiamine (B1) | 0.76 |
| **Butyrate** | **0.73** |
| **Propionate** | **0.62** |

All three principal short-chain fatty acids — acetate, butyrate,
propionate — plus seven amino acids (mean Pearson r = 0.78), three
B-vitamins (mean r = 0.75), and the tryptophan → indole gut-brain axis
are all directly recovered from shotgun species abundance alone.

Full per-compound metrics for all 1,780 KEGG compounds are shipped
with the companion manuscript as a supplementary table.

## Citation

If you use xFeed in your research, please cite:

> xFeed: predicting microbial cross-feeding flux from shotgun species abundance

## License

GPL-3.0 (academic / non-commercial) | commercial license available on request.
