"""Publication-ready visualizations for xFeed v1.1.0 predictions.

FluxMLP outputs a (n_samples, n_compounds=1780) matrix of per-sample
cross-feeding flux. Raw 1,780-wide tables are unreadable, so every
figure here either:
  (a) ranks compounds by mean flux and shows the top N, or
  (b) aggregates compounds into biologically meaningful categories.

All five figures are saved as PNG into an ``images/`` subdirectory
next to the prediction TSV.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

from .config import KEGG_API, REQUEST_DELAY

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Style
# ═══════════════════════════════════════════════════════════════════════

_PAL = {
    "SCFA":        "#4477AA",
    "Amino acid":  "#228833",
    "Vitamin":     "#EE6677",
    "Sugar":       "#CCBB44",
    "Cofactor":    "#AA3377",
    "Nucleotide":  "#66CCEE",
    "Aromatic":    "#EE7733",
    "Other":       "#888888",
}


def _apply_style() -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ═══════════════════════════════════════════════════════════════════════
# KEGG compound name lookup (with on-disk cache)
# ═══════════════════════════════════════════════════════════════════════

def fetch_compound_names(
    compound_ids: list[str],
    cache_path: Path | None = None,
) -> dict[str, str]:
    """Fetch compound → human-readable name via KEGG REST API.

    Uses batched requests (10 per call) and an on-disk JSON cache to
    avoid re-querying across runs.
    """
    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    missing = [c for c in compound_ids if c not in cache]
    if not missing:
        return {c: cache.get(c, c) for c in compound_ids}

    logger.info("Fetching %d compound names from KEGG...", len(missing))
    batch_size = 10
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        query = "+".join(batch)
        try:
            resp = requests.get(f"{KEGG_API}/list/{query}", timeout=30)
            if resp.status_code == 200:
                for line in resp.text.strip().split("\n"):
                    parts = line.split("\t", 1)
                    if len(parts) >= 2:
                        cid = parts[0].replace("cpd:", "")
                        name = parts[1].split(";")[0].strip()
                        cache[cid] = name
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.warning("  name fetch failed for batch: %s", e)

    # Write cache back
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

    return {c: cache.get(c, c) for c in compound_ids}


# ═══════════════════════════════════════════════════════════════════════
# Curated compound categories (used by compound_categories.png)
# ═══════════════════════════════════════════════════════════════════════

_CATEGORY_MAP: dict[str, str] = {
    # SCFAs
    "C00033": "SCFA",  # Acetate
    "C00246": "SCFA",  # Butyrate
    "C00163": "SCFA",  # Propionate
    "C00186": "SCFA",  # L-Lactate
    "C00256": "SCFA",  # R-Lactate
    "C00042": "SCFA",  # Succinate
    "C00058": "SCFA",  # Formate
    # Amino acids (20 standard)
    "C00025": "Amino acid",  # Glutamate
    "C00041": "Amino acid",  # Alanine
    "C00037": "Amino acid",  # Glycine
    "C00049": "Amino acid",  # Aspartate
    "C00064": "Amino acid",  # Glutamine
    "C00065": "Amino acid",  # Serine
    "C00073": "Amino acid",  # Methionine
    "C00078": "Amino acid",  # Tryptophan
    "C00079": "Amino acid",  # Phenylalanine
    "C00047": "Amino acid",  # Lysine
    "C00097": "Amino acid",  # Cysteine
    "C00123": "Amino acid",  # Leucine
    "C00183": "Amino acid",  # Valine
    "C00407": "Amino acid",  # Isoleucine
    "C00188": "Amino acid",  # Threonine
    "C00148": "Amino acid",  # Proline
    "C00135": "Amino acid",  # Histidine
    "C00152": "Amino acid",  # Asparagine
    "C00082": "Amino acid",  # Tyrosine
    "C00062": "Amino acid",  # Arginine
    "C00099": "Amino acid",  # β-Alanine
    # Vitamins
    "C00378": "Vitamin",  # Thiamine (B1)
    "C00255": "Vitamin",  # Riboflavin (B2)
    "C00253": "Vitamin",  # Niacin (B3)
    "C00864": "Vitamin",  # Pantothenate (B5)
    "C00314": "Vitamin",  # Pyridoxine (B6)
    "C00120": "Vitamin",  # Biotin (B7)
    "C00504": "Vitamin",  # Folate (B9)
    "C00176": "Vitamin",  # Cobalamin (B12)
    # Sugars
    "C00031": "Sugar",  # D-Glucose
    "C00095": "Sugar",  # D-Fructose
    "C00159": "Sugar",  # D-Mannose
    "C00124": "Sugar",  # D-Galactose
    "C00208": "Sugar",  # Maltose
    "C00089": "Sugar",  # Sucrose
    # Aromatic / gut-brain
    "C00463": "Aromatic",  # Indole
    "C00108": "Aromatic",  # Anthranilate
    "C00254": "Aromatic",  # Prephenate
    # Central cofactors
    "C00024": "Cofactor",  # Acetyl-CoA
    "C00100": "Cofactor",  # Propanoyl-CoA
    "C00136": "Cofactor",  # Butanoyl-CoA
    "C00170": "Cofactor",  # Methylmalonyl-CoA
    "C00091": "Cofactor",  # Succinyl-CoA
    "C00083": "Cofactor",  # Malonyl-CoA
}


def _assign_category(compound_id: str) -> str:
    return _CATEGORY_MAP.get(compound_id, "Other")


# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — Top 20 compounds by mean flux
# ═══════════════════════════════════════════════════════════════════════

# Minimum number of samples required for PCA embedding.  PCA on a
# single sample is undefined, and below this cap the projection is
# not informative.
_PCA_MIN_SAMPLES = 3

# Maximum number of characters to show in a compound-name row label
# before truncating with an ellipsis. Tuned so the widest label fits
# inside the figure's left margin without overflowing.
_LABEL_MAX_CHARS = 30


def _truncate(name: str, max_chars: int = _LABEL_MAX_CHARS) -> str:
    """Return ``name`` trimmed to ``max_chars`` characters with a
    trailing ellipsis if the original was longer. Preserves the full
    string if it already fits.
    """
    if len(name) <= max_chars:
        return name
    return name[: max_chars - 1].rstrip() + "…"


def _fig_top_compounds(
    df: pd.DataFrame,
    out_path: Path,
    top_n: int,
) -> None:
    """df: (compound_id, name, mean_flux) sorted by mean_flux descending.

    ``mean_flux`` is already in ``log1p`` space (i.e. ``ln(1 + raw_flux)``);
    the x-axis is labelled with that exact formula to avoid the common
    confusion that "log1p" looks like "log base 1 of p".
    """
    top = df.head(top_n).copy()

    def _row_label(r: pd.Series) -> str:
        """Two-line label: truncated readable name, then the KEGG ID."""
        name = _truncate(str(r["name"]))
        if r["name"] != r["compound_id"]:
            return f"{name}\n({r['compound_id']})"
        return r["compound_id"]

    top["label"] = top.apply(_row_label, axis=1)
    top["category"] = top["compound_id"].map(_assign_category)
    colors = [_PAL.get(c, _PAL["Other"]) for c in top["category"]]

    fig, ax = plt.subplots(figsize=(10.5, max(4, 0.36 * top_n + 1)))
    y = np.arange(len(top))
    ax.barh(y, top["mean_flux"], color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(top["label"], fontsize=8)
    ax.invert_yaxis()
    # Mathematically unambiguous label — ln(1 + x) is the natural log of
    # one plus the raw count, which is what np.log1p computes.
    ax.set_xlabel(r"Mean predicted flux  $\ln(1 + \mathrm{flux})$")
    ax.set_title(
        f"Top {top_n} cross-feedable compounds",
        fontsize=12, fontweight="bold", loc="left",
    )
    ax.grid(True, axis="x", alpha=0.25)
    sns.despine(ax=ax)

    # Category legend — show only categories that appear in the shown
    # bars, placed OUTSIDE the plotting area on the right so it cannot
    # overlap any bar or numeric annotation.
    from matplotlib.patches import Patch
    cat_counts: dict[str, int] = {}
    for c in top["category"]:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    cat_order = [c for c in _PAL if c in cat_counts]  # preserve palette order
    legend_handles = [
        Patch(
            facecolor=_PAL[c], edgecolor="#333",
            label=f"{c} ({cat_counts[c]})",
        )
        for c in cat_order
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Category",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8, title_fontsize=9,
            frameon=True, framealpha=0.9, edgecolor="#cccccc",
        )
        # Reserve right-margin space for the legend so tight_layout
        # does not crop it away.
        fig.subplots_adjust(right=0.80)

    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0)) if legend_handles \
        else fig.tight_layout()
    fig.savefig(out_path, facecolor="white", dpi=200)
    plt.close(fig)
    logger.info("  saved %s", out_path.name)


# ═══════════════════════════════════════════════════════════════════════
# Figure: Most VARIABLE compounds across samples (CV-ranked strip plot)
# ═══════════════════════════════════════════════════════════════════════

def _fig_variable_compounds(
    flux_log: np.ndarray,                # (n_samples, n_compounds) — log1p
    flux_raw: np.ndarray,                # (n_samples, n_compounds) — raw
    sample_ids: list[str],
    compound_list: list[str],
    compound_names: dict[str, str],
    out_path: Path,
    top_n: int = 20,
) -> None:
    """Horizontal strip plot of the ``top_n`` most-variable compounds.

    "Variable" is ranked by coefficient of variation (std / mean) in
    the RAW flux space so that relative differences between samples —
    not the absolute scale — drive the ranking. Each row shows the full
    per-sample distribution as jittered dots on the log1p axis, with
    the cohort mean marked as a short vertical bar for reference.

    This figure is the sample-level counterpart to
    ``_fig_top_compounds`` (which ranks by mean flux and therefore
    always highlights universal metabolites).
    """
    n_samples, n_compounds = flux_raw.shape
    if n_samples < 2:
        logger.warning(
            "  variable_compounds: need ≥ 2 samples, have %d — skipping",
            n_samples,
        )
        return

    mean = flux_raw.mean(axis=0)
    std = flux_raw.std(axis=0, ddof=0)
    # Coefficient of variation — zero-mean compounds (std == 0 too) are
    # filtered by the `valid` mask; near-zero mean with positive std
    # produces very high CV, which is what we want.
    valid = (mean > 0) & (std > 0)
    if valid.sum() < top_n:
        logger.warning(
            "  variable_compounds: only %d variable compounds — showing all",
            int(valid.sum()),
        )
    cv = np.where(valid, std / np.maximum(mean, 1e-12), 0.0)
    order = np.argsort(-cv)
    top_idx = order[: min(top_n, int(valid.sum()))]
    if len(top_idx) == 0:
        logger.warning("  variable_compounds: no variable compounds — skipping")
        return

    # Per-compound arrays in the display order
    top_ids = [compound_list[i] for i in top_idx]
    top_log = flux_log[:, top_idx]                 # (n_samples, k)
    top_cv = cv[top_idx]
    categories = [_assign_category(c) for c in top_ids]
    row_colors = [_PAL.get(c, _PAL["Other"]) for c in categories]

    def _row_label(i: int, cid: str) -> str:
        name = compound_names.get(cid, cid) or cid
        name = _truncate(name)
        return f"{name}\n({cid})" if name != cid else cid

    labels = [_row_label(i, cid) for i, cid in enumerate(top_ids)]

    fig, ax = plt.subplots(
        figsize=(11, max(4.5, 0.42 * len(top_ids) + 1)),
    )
    y = np.arange(len(top_ids))

    # Jitter the samples along y so overlapping points stay visible.
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.18, 0.18, size=(n_samples,))
    for j, yi in enumerate(y):
        xs = top_log[:, j]
        ax.scatter(
            xs, np.full_like(xs, yi) + jitter,
            s=28, color=row_colors[j],
            edgecolor="#333", linewidth=0.4,
            alpha=0.85, zorder=3,
        )
        # Short cohort-mean marker — a small vertical tick
        mean_log = float(xs.mean())
        ax.plot(
            [mean_log, mean_log], [yi - 0.28, yi + 0.28],
            color="#111", lw=1.3, zorder=4,
        )

    # CV annotation on the right
    xmax_disp = float(top_log.max()) * 1.05 if top_log.size else 1.0
    for yi, c in zip(y, top_cv):
        ax.text(
            xmax_disp, yi, f"CV = {c:.2f}",
            va="center", ha="right", fontsize=7, color="#555",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"Predicted flux  $\ln(1 + \mathrm{flux})$")
    ax.set_title(
        f"Top {len(top_ids)} most variable compounds  "
        f"(n = {n_samples} samples, ranked by CV)",
        fontsize=12, fontweight="bold", loc="left",
    )
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_ylim(-0.8, len(top_ids) - 0.2)
    sns.despine(ax=ax)

    # Category legend — outside the plot, same style as _fig_top_compounds
    from matplotlib.patches import Patch
    cat_counts: dict[str, int] = {}
    for c in categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    cat_order = [c for c in _PAL if c in cat_counts]
    legend_handles = [
        Patch(facecolor=_PAL[c], edgecolor="#333",
              label=f"{c} ({cat_counts[c]})")
        for c in cat_order
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Category",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8, title_fontsize=9,
            frameon=True, framealpha=0.9, edgecolor="#cccccc",
        )
        fig.subplots_adjust(right=0.80)
        fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    else:
        fig.tight_layout()

    fig.savefig(out_path, facecolor="white", dpi=200)
    plt.close(fig)
    logger.info("  saved %s", out_path.name)


# ═══════════════════════════════════════════════════════════════════════
# Figure: Flux-space PCA (2D sample embedding via SVD)
# ═══════════════════════════════════════════════════════════════════════

def _fig_flux_pca(
    flux_log: np.ndarray,                 # (n_samples, n_compounds)
    sample_ids: list[str],
    out_path: Path,
) -> None:
    """Project samples onto the first two principal components of the
    predicted log-flux space and show the resulting 2-D embedding.

    PCA is computed via centred SVD (no sklearn dependency). Points are
    coloured by total log-flux as a proxy for overall community
    metabolic activity. Sample IDs are annotated for the 10 most
    extreme points so the scatter stays readable when many samples
    cluster together.
    """
    n_samples = flux_log.shape[0]
    if n_samples < _PCA_MIN_SAMPLES:
        logger.warning(
            "  flux_pca: need ≥ %d samples, have %d — skipping",
            _PCA_MIN_SAMPLES, n_samples,
        )
        return

    # Centre each compound column and run a thin SVD
    X = flux_log - flux_log.mean(axis=0, keepdims=True)
    try:
        U, S, _Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.warning("  flux_pca: SVD failed — skipping")
        return

    # Principal scores and explained-variance ratios
    scores = U * S                     # (n_samples, min(n_samples, n_compounds))
    var_ratio = (S ** 2) / float(np.sum(S ** 2) if np.sum(S ** 2) > 0 else 1.0)
    pc1 = scores[:, 0] if scores.shape[1] >= 1 else np.zeros(n_samples)
    pc2 = scores[:, 1] if scores.shape[1] >= 2 else np.zeros(n_samples)
    ev1 = var_ratio[0] * 100 if len(var_ratio) >= 1 else 0.0
    ev2 = var_ratio[1] * 100 if len(var_ratio) >= 2 else 0.0

    # Point colour = total log-flux (community "metabolic activity")
    totals = flux_log.sum(axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sc = ax.scatter(
        pc1, pc2, c=totals, cmap="viridis",
        s=70, edgecolor="#222", linewidth=0.6, alpha=0.92,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Total predicted log-flux (per sample)")

    # Annotate up to 10 most-extreme samples (by L2 distance from origin)
    dists = np.sqrt(pc1 ** 2 + pc2 ** 2)
    if n_samples <= 12:
        label_idx = np.arange(n_samples)
    else:
        label_idx = np.argsort(-dists)[:10]
    for i in label_idx:
        ax.annotate(
            sample_ids[i],
            xy=(pc1[i], pc2[i]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="#111",
        )

    ax.axhline(0, color="#888", lw=0.6, ls="--")
    ax.axvline(0, color="#888", lw=0.6, ls="--")
    ax.set_xlabel(f"PC1 ({ev1:.1f} % variance)")
    ax.set_ylabel(f"PC2 ({ev2:.1f} % variance)")
    ax.set_title(
        f"Flux-space PCA  (n = {n_samples} samples, "
        f"{flux_log.shape[1]} compounds)",
        fontsize=12, fontweight="bold", loc="left",
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", dpi=200)
    plt.close(fig)
    logger.info("  saved %s", out_path.name)


# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — Sample × top-20 compound heatmap
# ═══════════════════════════════════════════════════════════════════════

def _build_pair_interaction_matrix(
    species_subset: list[str],
    species_caps: dict[str, dict[str, set[str]]],
) -> np.ndarray:
    """Return a symmetric (n × n) int matrix of cross-feeding interaction
    counts for ``species_subset``.

    ``M[i, j]`` = number of compounds where species_i produces and
    species_j consumes (directional), counted both directions so M is
    symmetric. Diagonal is set to zero (no self cross-feeding).
    """
    n = len(species_subset)
    M = np.zeros((n, n), dtype=np.int64)
    caps_list = [
        (species_caps.get(sp, {}).get("produces", set()),
         species_caps.get(sp, {}).get("consumes", set()))
        for sp in species_subset
    ]
    for i in range(n):
        p_i, c_i = caps_list[i]
        if not p_i and not c_i:
            continue
        for j in range(n):
            if i == j:
                continue
            _p_j, c_j = caps_list[j]
            # number of compounds where i → j (producer → consumer)
            M[i, j] = len(p_i & c_j)
    # Symmetrize: add ij and ji so the heatmap is undirected
    return M + M.T


def _select_top_species_by_degree(
    species_caps: dict[str, dict[str, set[str]]],
    top_n: int,
    abundance: np.ndarray | None = None,
    species_list: list[str] | None = None,
    present_only: bool = True,
) -> list[str]:
    """Pick the ``top_n`` species with the highest cross-feeding degree.

    Degree is approximated by ``|produces| + |consumes|`` — the number
    of metabolites the species can participate in. If ``abundance`` /
    ``species_list`` are provided and ``present_only`` is True, only
    species present in the input samples (mean abundance > 0) are
    considered.
    """
    candidates = list(species_caps.keys())
    if present_only and abundance is not None and species_list is not None:
        mean_abd = abundance.mean(axis=0)
        present = {
            sp for sp, a in zip(species_list, mean_abd) if a > 0.0
        }
        candidates = [sp for sp in candidates if sp in present]

    def _degree(sp: str) -> int:
        caps = species_caps.get(sp, {})
        return len(caps.get("produces", set())) + len(caps.get("consumes", set()))

    candidates.sort(key=_degree, reverse=True)
    return candidates[:top_n]


def _fig_heatmap(
    out_path: Path,
    species_caps: dict[str, dict[str, set[str]]],
    top_n: int = 20,
    abundance: np.ndarray | None = None,
    species_list: list[str] | None = None,
) -> None:
    """Species × species cross-feeding interaction count heatmap.

    Matches ``examples/heatmap.png`` — the y and x axes are the top-N
    species selected by total cross-feeding degree; cell (i, j) is the
    number of distinct compounds exchanged between species i and j in
    either direction; colormap is YlOrRd; the colorbar is labelled
    "interaction count".
    """
    species_subset = _select_top_species_by_degree(
        species_caps, top_n,
        abundance=abundance, species_list=species_list,
    )
    if len(species_subset) < 2:
        logger.warning("  heatmap: not enough species — skipping")
        return

    M = _build_pair_interaction_matrix(species_subset, species_caps)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        M,
        cmap="YlOrRd",
        xticklabels=species_subset,
        yticklabels=species_subset,
        cbar_kws={"label": "interaction count"},
        linewidths=0.4, linecolor="#f0f0f0",
        square=False,
        ax=ax,
    )
    ax.set_title(
        f"Species interaction frequency (top {len(species_subset)})",
        fontsize=12, fontweight="bold",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", dpi=200)
    plt.close(fig)
    logger.info("  saved %s", out_path.name)


# ═══════════════════════════════════════════════════════════════════════
# Figure: Most frequently exchanged compounds (horizontal bar)
# ═══════════════════════════════════════════════════════════════════════

def _fig_compound_distribution(
    out_path: Path,
    species_caps: dict[str, dict[str, set[str]]],
    compound_names: dict[str, str],
    top_n: int = 15,
    abundance: np.ndarray | None = None,
    species_list: list[str] | None = None,
) -> None:
    """Top-N compounds ranked by the number of producer-consumer species
    pairs capable of exchanging them.

    Matches ``examples/compound_distribution.png``. For each compound C
    the value is ``|producers(C)| × |consumers(C)|`` — the same rule used
    to build xFeed's flux labels, here summed across the species pool.
    """
    # Select species pool: only species observed in the input samples
    # if abundance is provided, otherwise the full catalog.
    pool: list[str]
    if abundance is not None and species_list is not None:
        mean_abd = abundance.mean(axis=0)
        pool = [
            sp for sp, a in zip(species_list, mean_abd)
            if a > 0.0 and sp in species_caps
        ]
    else:
        pool = list(species_caps.keys())
    if not pool:
        logger.warning("  compound_distribution: empty species pool — skipping")
        return

    # Gather the producer / consumer counts per compound
    from collections import Counter
    producers: Counter[str] = Counter()
    consumers: Counter[str] = Counter()
    for sp in pool:
        caps = species_caps.get(sp, {})
        for cid in caps.get("produces", ()):
            producers[cid] += 1
        for cid in caps.get("consumes", ()):
            consumers[cid] += 1

    all_compounds = set(producers) & set(consumers)
    if not all_compounds:
        logger.warning("  compound_distribution: no cross-feedable compounds — skipping")
        return

    scores = {
        cid: producers[cid] * consumers[cid] for cid in all_compounds
    }
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    cids = [c for c, _ in ranked]
    values = [s for _, s in ranked]
    labels = [
        (compound_names.get(c, c) or c)[:38] for c in cids
    ]

    cmap = plt.get_cmap("tab20", len(cids))
    colors = [cmap(i) for i in range(len(cids))]

    fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.35 * len(cids) + 1.5)))
    y = np.arange(len(cids))
    ax.barh(y, values, color=colors, edgecolor="#333", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of cross-feeding interactions")
    ax.set_title(
        "Most frequently exchanged compounds",
        fontsize=13, fontweight="bold",
    )
    # Value labels at bar ends
    xmax = max(values) if values else 1
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.01, yi, f"{v:,}", va="center", fontsize=8)
    ax.set_xlim(0, xmax * 1.12)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", dpi=200)
    plt.close(fig)
    logger.info("  saved %s", out_path.name)


# ═══════════════════════════════════════════════════════════════════════
# Figure: Species by Cross-feeding Degree (horizontal bar)
# ═══════════════════════════════════════════════════════════════════════

def _fig_crossfeeding_degree(
    out_path: Path,
    species_caps: dict[str, dict[str, set[str]]],
    compound_names: dict[str, str],
    top_n: int = 20,
    abundance: np.ndarray | None = None,
    species_list: list[str] | None = None,
) -> None:
    """Top-N species ranked by total cross-feeding interactions,
    colored by each species' top (most-connecting) compound.

    Matches ``examples/crossfeeding_degree.png``. For each species S we
    compute (i) a total interaction count — the sum over every compound
    C the species produces or consumes of the number of complementary
    species in the pool — and (ii) its top compound, the compound
    contributing the largest share of that count. Bars are sized by the
    total and colored by the top compound.
    """
    from collections import Counter

    # Species pool (present in samples if abundance is provided)
    if abundance is not None and species_list is not None:
        mean_abd = abundance.mean(axis=0)
        pool = [
            sp for sp, a in zip(species_list, mean_abd)
            if a > 0.0 and sp in species_caps
        ]
    else:
        pool = list(species_caps.keys())
    if not pool:
        logger.warning("  crossfeeding_degree: empty species pool — skipping")
        return

    # Global producer / consumer counts
    producers: Counter[str] = Counter()
    consumers: Counter[str] = Counter()
    for sp in pool:
        caps = species_caps.get(sp, {})
        for cid in caps.get("produces", ()):
            producers[cid] += 1
        for cid in caps.get("consumes", ()):
            consumers[cid] += 1

    species_stats: list[tuple[str, int, str]] = []
    for sp in pool:
        caps = species_caps.get(sp, {})
        p_set = caps.get("produces", set())
        c_set = caps.get("consumes", set())
        total = 0
        contrib: dict[str, int] = {}
        for cid in p_set:
            # species S produces C → matches to every consumer of C except S
            k = consumers.get(cid, 0) - (1 if cid in c_set else 0)
            if k > 0:
                total += k
                contrib[cid] = contrib.get(cid, 0) + k
        for cid in c_set:
            # species S consumes C → matches to every producer of C except S
            k = producers.get(cid, 0) - (1 if cid in p_set else 0)
            if k > 0:
                total += k
                contrib[cid] = contrib.get(cid, 0) + k
        if total > 0 and contrib:
            top_cid = max(contrib.items(), key=lambda x: x[1])[0]
            species_stats.append((sp, total, top_cid))

    species_stats.sort(key=lambda x: -x[1])
    species_stats = species_stats[:top_n]
    if not species_stats:
        logger.warning("  crossfeeding_degree: no eligible species — skipping")
        return

    sp_names = [s for s, _, _ in species_stats]
    totals = [t for _, t, _ in species_stats]
    top_cids = [c for _, _, c in species_stats]

    unique_top = sorted(set(top_cids))
    cmap = plt.get_cmap("Set2", max(len(unique_top), 3))
    cid_color = {cid: cmap(i) for i, cid in enumerate(unique_top)}
    bar_colors = [cid_color[c] for c in top_cids]

    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.36 * len(sp_names) + 1.5)))
    y = np.arange(len(sp_names))
    ax.barh(y, totals, color=bar_colors, edgecolor="#333", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(sp_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Total interactions (producer + consumer)")
    ax.set_title(
        "Species by Cross-feeding Degree",
        fontsize=13, fontweight="bold",
    )

    # Inline italic labels for the top compound on each bar
    xmax = max(totals) if totals else 1
    for yi, t, cid in zip(y, totals, top_cids):
        name = (compound_names.get(cid, cid) or cid)[:40]
        ax.text(
            t + xmax * 0.01, yi, name,
            va="center", fontsize=7, style="italic", color="#555",
        )
    ax.set_xlim(0, xmax * 1.22)

    # Legend for top-compound colors (once per unique compound)
    from matplotlib.patches import Patch
    handles = [
        Patch(
            facecolor=cid_color[cid], edgecolor="#333",
            label=(compound_names.get(cid, cid) or cid)[:30],
        )
        for cid in unique_top
    ]
    ax.legend(
        handles=handles, title="Top compound",
        loc="lower right", fontsize=8, title_fontsize=9,
        frameon=True, framealpha=0.9, edgecolor="#cccccc",
    )

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", dpi=200)
    plt.close(fig)
    logger.info("  saved %s", out_path.name)


# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — Flux density per sample
# ═══════════════════════════════════════════════════════════════════════

def _fig_flux_density(
    flux_raw: np.ndarray,
    out_path: Path,
) -> None:
    total_per_sample = flux_raw.sum(axis=1)
    sorted_totals = np.sort(total_per_sample)[::-1]
    n_samples = len(total_per_sample)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # Adaptive bin count — matplotlib raises if there are more requested
    # bins than distinct finite values in the data (e.g. 1 sample).
    n_bins = max(1, min(40, n_samples))
    if n_samples == 1:
        # Single bar centred at the one value; hist with a single bar
        # looks silly so we draw a bar chart instead.
        axes[0].bar(
            [0], total_per_sample, color="#4477AA",
            edgecolor="#333", linewidth=0.5, alpha=0.85, width=0.6,
        )
        axes[0].set_xticks([0])
        axes[0].set_xticklabels(["sample 1"])
    else:
        axes[0].hist(
            total_per_sample, bins=n_bins, color="#4477AA",
            edgecolor="#333", linewidth=0.5, alpha=0.85,
        )
    axes[0].set_xlabel("Total predicted flux / sample")
    axes[0].set_ylabel("Sample count")
    axes[0].set_title("(a) Flux distribution", fontsize=11, loc="left")
    axes[0].grid(True, axis="y", alpha=0.2)
    sns.despine(ax=axes[0])

    axes[1].plot(
        np.arange(1, len(sorted_totals) + 1),
        sorted_totals, color="#AA3377", lw=1.5,
    )
    axes[1].fill_between(
        np.arange(1, len(sorted_totals) + 1),
        sorted_totals, color="#AA3377", alpha=0.2,
    )
    axes[1].set_xlabel("Sample rank")
    axes[1].set_ylabel("Total flux (descending)")
    axes[1].set_title("(b) Samples ranked by total flux",
                       fontsize=11, loc="left")
    axes[1].grid(True, alpha=0.2)
    sns.despine(ax=axes[1])

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    logger.info("  saved %s", out_path.name)


# ═══════════════════════════════════════════════════════════════════════
# Figure 4 — Per-sample top-20 compound composition (stacked bar)
# ═══════════════════════════════════════════════════════════════════════

def _fig_composition(
    flux_raw: np.ndarray,
    compound_list: list[str],
    compound_names: dict[str, str],
    sample_ids: list[str],
    out_path: Path,
    top_n: int,
    max_samples: int = 40,
) -> None:
    means = flux_raw.mean(axis=0)
    top_idx = np.argsort(-means)[:top_n]
    top_ids = [compound_list[i] for i in top_idx]
    top_labels = [
        compound_names.get(cid, cid)[:14] for cid in top_ids
    ]

    m_top = flux_raw[:, top_idx]                      # (n_samples, top_n)
    m_rest_total = flux_raw.sum(axis=1) - m_top.sum(axis=1)

    n_samples = len(sample_ids)
    if n_samples > max_samples:
        rng = np.random.default_rng(0)
        sel = rng.choice(n_samples, max_samples, replace=False)
        sel.sort()
        m_top = m_top[sel]
        m_rest_total = m_rest_total[sel]
        shown_labels = [sample_ids[i] for i in sel]
        suffix = f" (subsampled: {max_samples} of {n_samples})"
    else:
        shown_labels = sample_ids
        suffix = ""

    cmap = plt.get_cmap("tab20", top_n)
    colors = [cmap(i) for i in range(top_n)]

    fig, ax = plt.subplots(figsize=(max(7, 0.22 * len(shown_labels) + 3), 5.5))
    x = np.arange(len(shown_labels))
    bottom = np.zeros(len(shown_labels))
    for i in range(top_n):
        ax.bar(x, m_top[:, i], bottom=bottom, color=colors[i],
               label=top_labels[i], width=0.85, edgecolor="none")
        bottom += m_top[:, i]
    # Other
    ax.bar(x, m_rest_total, bottom=bottom, color="#dddddd",
           label="Other", width=0.85, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(shown_labels, rotation=90, fontsize=6)
    ax.set_ylabel("Predicted flux")
    ax.set_title(
        f"Per-sample compound composition  (top {top_n} + Other){suffix}",
        fontsize=12, fontweight="bold", loc="left",
    )
    ax.legend(
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=7, frameon=False, ncol=1,
    )
    ax.grid(True, axis="y", alpha=0.2)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    logger.info("  saved %s", out_path.name)


# ═══════════════════════════════════════════════════════════════════════
# Figure: Species-as-nodes, compound-as-edges cross-feeding network
# ═══════════════════════════════════════════════════════════════════════

def _fig_species_network(
    abundance: np.ndarray,
    species_list: list[str],
    flux_raw: np.ndarray,
    compound_list: list[str],
    compound_names: dict[str, str],
    species_caps: dict[str, dict[str, set[str]]],
    out_path: Path,
    top_n_compounds: int = 10,
    top_n_species: int = 80,
) -> None:
    """Draw a dense cross-feeding hairball figure matching examples/network.png.

    - Nodes: up to `top_n_species` most abundant species across the input
      samples that also participate in at least one producer → consumer
      edge for the selected top compounds.
    - Edges: for each of the top `top_n_compounds` compounds by predicted
      mean flux, every producer species is connected to every consumer
      species among the selected nodes. Edge color = compound; edge
      width ∝ mean predicted flux of that compound.
    - Node color = genus (first word of the binomial name).
    - Node labels = full species name, drawn with a thin white bbox so
      that they remain readable against dense edge bundles.
    - Legends (Genus and Compound) are placed *inside* the axes to avoid
      shrinking the plotting area, matching the published example.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning(
            "networkx not available — skipping species network figure. "
            "Install with `pip install networkx` to enable it."
        )
        return

    # 1. Top compounds by mean predicted flux
    mean_flux = flux_raw.mean(axis=0)
    top_compound_idx = np.argsort(-mean_flux)[:top_n_compounds]
    top_compounds = [compound_list[i] for i in top_compound_idx]

    # 2. Top species by mean abundance, keeping ONLY species that are
    # actually present in the input samples (mean abundance > 0). This
    # avoids drawing "phantom" nodes that were in the profile catalog
    # but never observed in the cohort.
    mean_abd = abundance.mean(axis=0)
    present_mask = mean_abd > 0.0
    # Sort descending by mean abundance, then slice to top_n_species AMONG
    # present species only.
    present_order = np.argsort(-mean_abd)
    present_order = present_order[present_mask[present_order]]
    top_sp_idx = present_order[:top_n_species]
    top_species = [species_list[i] for i in top_sp_idx]

    # 3. Build edges (producer → consumer for each top compound)
    edges: list[tuple[str, str, str, float]] = []
    for ci, cid in zip(top_compound_idx, top_compounds):
        weight = float(mean_flux[ci])
        producers = [
            sp for sp in top_species
            if cid in species_caps.get(sp, {}).get("produces", set())
        ]
        consumers = [
            sp for sp in top_species
            if cid in species_caps.get(sp, {}).get("consumes", set())
        ]
        for p in producers:
            for c in consumers:
                if p == c:
                    continue
                edges.append((p, c, cid, weight))

    if not edges:
        logger.warning(
            "  species network: no producer–consumer edges among top %d "
            "species for top %d compounds — skipping",
            top_n_species, top_n_compounds,
        )
        return

    # Keep only species that appear in at least one edge
    edge_species = {p for p, *_ in edges} | {c for _, c, *_ in edges}
    active_species = [sp for sp in top_species if sp in edge_species]
    abd_lookup = {sp: mean_abd[top_sp_idx[i]]
                  for i, sp in enumerate(top_species) if sp in edge_species}

    # 4. Assign genus color (nodes)
    def _genus(name: str) -> str:
        return name.split()[0] if name else "Unknown"

    genera = sorted({_genus(sp) for sp in active_species})
    # Use a larger, more distinctive palette for many genera
    if len(genera) <= 20:
        genus_cmap = plt.get_cmap("tab20", max(len(genera), 1))
    else:
        genus_cmap = plt.get_cmap("hsv", max(len(genera), 1))
    genus_color = {g: genus_cmap(i) for i, g in enumerate(genera)}

    # 5. Assign compound color (edges)
    compound_cmap = plt.get_cmap("tab10", max(top_n_compounds, 1))
    compound_color = {
        cid: compound_cmap(i) for i, cid in enumerate(top_compounds)
    }

    # 6. Build graph
    G = nx.DiGraph()
    for sp in active_species:
        G.add_node(
            sp,
            abundance=float(abd_lookup.get(sp, 0.0)),
            genus=_genus(sp),
        )
    for p, c, cid, w in edges:
        if G.has_edge(p, c):
            G[p][c]["weight"] += w
            G[p][c]["compounds"].append(cid)
        else:
            G.add_edge(p, c, weight=w, compounds=[cid])

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_genera = len(genera)

    # 7. Layout — vanilla Fruchterman–Reingold (unweighted) produces the
    # classic organic hairball. We leave `weight=None` so huge flux values
    # do not collapse the graph into a single blob.
    pos = nx.spring_layout(
        G, seed=42,
        k=1.1 / max(n_nodes, 1) ** 0.5,
        iterations=250,
        weight=None,
    )

    # Center and rescale positions into [-1, 1] with a 5 % margin so
    # labels and outer tendrils do not get clipped.
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    x_mid, y_mid = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    span = max(xs.max() - xs.min(), ys.max() - ys.min()) or 1.0
    pos = {
        k: ((v[0] - x_mid) / span * 2.0, (v[1] - y_mid) / span * 2.0)
        for k, v in pos.items()
    }

    # 8. Figure — aesthetic defaults, square canvas, white margins
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    # --- Edges: color by dominant compound, width by flux, very low alpha
    # so the hub is readable and outer tendrils stay visible.
    max_edge_weight = max(
        (d["weight"] for _, _, d in G.edges(data=True)), default=1.0,
    )
    # Draw edges in order of weight (thinnest first) so the thick ones
    # land on top — more depth, less muddy middle.
    edges_sorted = sorted(
        G.edges(data=True), key=lambda e: e[2]["weight"],
    )
    for u, v, d in edges_sorted:
        dominant_cid = d["compounds"][0]
        color = compound_color.get(dominant_cid, (0.6, 0.6, 0.6))
        w_norm = d["weight"] / max_edge_weight
        lw = 0.3 + 1.6 * w_norm
        alpha = 0.15 + 0.35 * w_norm
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color=[color], width=lw,
            alpha=alpha,
            arrows=False,
            ax=ax,
        )

    # --- Nodes: size by abundance, color by genus, soft outline
    max_abd = max(abd_lookup.values(), default=1.0) or 1.0
    node_colors = [genus_color[_genus(sp)] for sp in G.nodes()]
    node_sizes = [
        220 + 1400 * (G.nodes[sp]["abundance"] / max_abd)
        for sp in G.nodes()
    ]
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white", linewidths=1.2,
        ax=ax,
    )

    # --- Node labels: take the top-K species by (visual) abundance so
    # every large circle on the plot is named, plus the top-M by degree
    # so well-connected hubs (even if small) also get a label. Union of
    # these two sets.
    by_abd = sorted(
        G.nodes(), key=lambda n: G.nodes[n]["abundance"], reverse=True,
    )
    by_deg = sorted(
        G.nodes(), key=lambda n: G.degree(n, weight="weight"), reverse=True,
    )
    label_set = set(by_abd[: min(20, n_nodes)]) | set(by_deg[: min(10, n_nodes)])
    for sp, (x, y) in pos.items():
        if sp not in label_set:
            continue
        ax.text(
            x, y, sp,
            ha="center", va="center",
            fontsize=7, color="#111111",
            bbox=dict(
                facecolor="white", edgecolor="#555555",
                boxstyle="round,pad=0.25", linewidth=0.6,
                alpha=0.92,
            ),
            zorder=10,
        )

    # --- Title at top-center
    n_samples = int(abundance.shape[0]) if abundance is not None else 0
    ax.set_title(
        f"Cross-feeding network  "
        f"({n_samples} samples  ·  {n_nodes} species  ·  "
        f"{n_genera} genera  ·  {n_edges} edges)",
        fontsize=14, fontweight="bold", pad=14,
    )

    # --- Two legends inside the axes (top-left: genus, bottom-left: compound)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Genus legend — show top 12 genera
    genus_counts: dict[str, int] = {}
    for sp in active_species:
        g = _genus(sp)
        genus_counts[g] = genus_counts.get(g, 0) + 1
    top_genera_list = sorted(
        genus_counts.items(), key=lambda x: -x[1],
    )[:12]
    genus_handles = [
        Patch(facecolor=genus_color[g], edgecolor="#333",
              label=f"{g} ({n})")
        for g, n in top_genera_list
    ]
    leg1 = ax.legend(
        handles=genus_handles,
        title="Genus (node)",
        loc="upper left",
        fontsize=9, title_fontsize=10, frameon=True,
        framealpha=0.92, edgecolor="#cccccc",
    )
    ax.add_artist(leg1)

    # Compound legend — names only, no "(C00033)"
    compound_handles = [
        Line2D(
            [0], [0],
            color=compound_color[cid], lw=3.0,
            label=(compound_names.get(cid, cid) or cid)[:28],
        )
        for cid in top_compounds
    ]
    ax.legend(
        handles=compound_handles,
        title="Compound (edge)",
        loc="lower left",
        fontsize=9, title_fontsize=10, frameon=True,
        framealpha=0.92, edgecolor="#cccccc",
    )

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", dpi=300)
    plt.close(fig)
    logger.info(
        "  saved %s  (%d species, %d genera, %d edges)",
        out_path.name, n_nodes, n_genera, n_edges,
    )


# ═══════════════════════════════════════════════════════════════════════
# Figure: Sankey (producer → compound → consumer)
# ═══════════════════════════════════════════════════════════════════════

def _fig_sankey(
    abundance: np.ndarray,
    species_list: list[str],
    flux_raw: np.ndarray,
    compound_list: list[str],
    compound_names: dict[str, str],
    species_caps: dict[str, dict[str, set[str]]],
    out_path: Path,
    top_n_compounds: int = 10,
    top_n_producers: int = 12,
    top_n_consumers: int = 12,
    focus_compounds: list[str] | None = None,
) -> None:
    """3-column Sankey diagram drawn entirely in matplotlib.

    Layout:
        LEFT (producers)  →  CENTER (compounds)  →  RIGHT (consumers)

    Weights:
        producer → compound width = flux × abundance(producer)
        compound → consumer width = flux × abundance(consumer)
    """
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    # 1. Pick compounds: focus_compounds if provided, else top-K by mean flux
    mean_flux = flux_raw.mean(axis=0)
    if focus_compounds:
        top_c_idx = np.array([
            compound_list.index(c) for c in focus_compounds
            if c in compound_list
        ])
    else:
        top_c_idx = np.argsort(-mean_flux)[:top_n_compounds]
    top_c = [(compound_list[i], float(mean_flux[i])) for i in top_c_idx]

    if not top_c:
        logger.warning("  sankey: no compounds to draw — skipping")
        return

    # 2. For each compound, rank producer & consumer species by mean abundance,
    #    and compute flow weights.
    mean_abd = abundance.mean(axis=0)
    abd_lookup = {sp: float(mean_abd[i]) for i, sp in enumerate(species_list)}

    producer_flows: dict[tuple[str, str], float] = {}
    consumer_flows: dict[tuple[str, str], float] = {}
    for cid, cflux in top_c:
        produces_sp = [
            sp for sp in species_list
            if cid in species_caps.get(sp, {}).get("produces", set())
        ]
        consumes_sp = [
            sp for sp in species_list
            if cid in species_caps.get(sp, {}).get("consumes", set())
        ]
        # Rank by abundance
        produces_sp = sorted(produces_sp, key=lambda s: -abd_lookup[s])[:top_n_producers]
        consumes_sp = sorted(consumes_sp, key=lambda s: -abd_lookup[s])[:top_n_consumers]
        # Distribute the compound's flux proportionally to abundance
        total_p_abd = sum(abd_lookup[s] for s in produces_sp) or 1.0
        total_c_abd = sum(abd_lookup[s] for s in consumes_sp) or 1.0
        for s in produces_sp:
            w = cflux * (abd_lookup[s] / total_p_abd)
            producer_flows[(s, cid)] = producer_flows.get((s, cid), 0.0) + w
        for s in consumes_sp:
            w = cflux * (abd_lookup[s] / total_c_abd)
            consumer_flows[(cid, s)] = consumer_flows.get((cid, s), 0.0) + w

    # 3. Deduplicate + order the left / center / right columns
    left_species = sorted(
        {p for p, _ in producer_flows.keys()},
        key=lambda s: -sum(w for (p, _), w in producer_flows.items() if p == s),
    )
    right_species = sorted(
        {c for _, c in consumer_flows.keys()},
        key=lambda s: -sum(w for (_, c), w in consumer_flows.items() if c == s),
    )
    center_compounds = [c for c, _ in top_c]

    if not left_species or not right_species:
        logger.warning("  sankey: empty producer or consumer set — skipping")
        return

    # 4. Compute node heights (proportional to total flow through the node)
    def _node_height(node_flows: dict, key_idx: int, node: str) -> float:
        return sum(w for key, w in node_flows.items() if key[key_idx] == node)

    left_heights = [_node_height(producer_flows, 0, s) for s in left_species]
    right_heights = [_node_height(consumer_flows, 1, s) for s in right_species]
    # Center node height = sum of incoming + outgoing / 2 (they should match)
    center_heights = []
    for cid in center_compounds:
        in_flow = sum(w for (_, c), w in producer_flows.items() if c == cid)
        out_flow = sum(w for (c, _), w in consumer_flows.items() if c == cid)
        center_heights.append(max(in_flow, out_flow))

    # 5. Normalize columns to share the same total height
    def _normalize(heights: list[float], total: float) -> list[float]:
        s = sum(heights) or 1.0
        return [h / s * total for h in heights]

    TOTAL_H = 10.0
    PAD = 0.25
    n_left = len(left_species)
    n_center = len(center_compounds)
    n_right = len(right_species)

    avail_left = TOTAL_H - PAD * (n_left - 1)
    avail_center = TOTAL_H - PAD * (n_center - 1)
    avail_right = TOTAL_H - PAD * (n_right - 1)

    left_h = _normalize(left_heights, avail_left)
    center_h = _normalize(center_heights, avail_center)
    right_h = _normalize(right_heights, avail_right)

    # 6. Compute y-positions (top of each node) in each column
    def _y_positions(heights: list[float]) -> list[tuple[float, float]]:
        """Return (y_top, y_bottom) for each node, top-down."""
        positions = []
        y = TOTAL_H
        for h in heights:
            positions.append((y, y - h))
            y -= h + PAD
        return positions

    left_pos = _y_positions(left_h)
    center_pos = _y_positions(center_h)
    right_pos = _y_positions(right_h)

    left_lookup = {s: left_pos[i] for i, s in enumerate(left_species)}
    center_lookup = {c: center_pos[i] for i, c in enumerate(center_compounds)}
    right_lookup = {s: right_pos[i] for i, s in enumerate(right_species)}

    # 7. Set up figure
    fig, ax = plt.subplots(figsize=(14, max(7, 0.28 * TOTAL_H + 6)))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, TOTAL_H + 1)
    ax.set_axis_off()

    NODE_W = 0.30
    X_LEFT = 1.2
    X_CENTER = 5.0
    X_RIGHT = 8.8

    # 8. Draw nodes as thin rectangles
    def _draw_node_column(
        column_x: float, positions, items,
        color_fn, fontsize: float,
        label_side: str,  # "left" or "right" of the rectangle
        draw_label: bool = True,
    ):
        for (y_top, y_bot), name in zip(positions, items):
            h = y_top - y_bot
            ax.add_patch(Rectangle(
                (column_x - NODE_W / 2, y_bot),
                NODE_W, h,
                facecolor=color_fn(name),
                edgecolor="#333", linewidth=0.6,
                zorder=5,
            ))
            # Center column is relabeled later with the readable compound
            # name — skip drawing the raw ID here so it does not bleed
            # through behind the final label box.
            if not draw_label:
                continue
            if label_side == "left":
                ax.text(
                    column_x - NODE_W / 2 - 0.1, (y_top + y_bot) / 2,
                    name, ha="right", va="center",
                    fontsize=fontsize, color="#111",
                )
            elif label_side == "right":
                ax.text(
                    column_x + NODE_W / 2 + 0.1, (y_top + y_bot) / 2,
                    name, ha="left", va="center",
                    fontsize=fontsize, color="#111",
                )
            else:
                ax.text(
                    column_x, y_top + 0.08,
                    name, ha="center", va="bottom",
                    fontsize=fontsize, color="#111", fontweight="bold",
                )

    # Genus palette for producer/consumer columns
    def _genus(name: str) -> str:
        return name.split()[0] if name else "Unknown"

    genera_all = sorted(
        {_genus(s) for s in left_species} | {_genus(s) for s in right_species}
    )
    genus_cmap = plt.get_cmap("tab20", max(len(genera_all), 1))
    genus_color = {g: genus_cmap(i) for i, g in enumerate(genera_all)}

    def _sp_color(sp: str):
        return genus_color[_genus(sp)]

    # Compound palette
    compound_cmap = plt.get_cmap("Set3", max(len(center_compounds), 1))
    compound_color = {
        cid: compound_cmap(i) for i, cid in enumerate(center_compounds)
    }

    def _c_color(cid: str):
        return compound_color[cid]

    _draw_node_column(
        X_LEFT, left_pos, left_species, _sp_color,
        fontsize=7, label_side="left",
    )
    _draw_node_column(
        X_CENTER, center_pos, center_compounds,
        lambda c: compound_color[c],
        fontsize=7.5, label_side="right",
        draw_label=False,  # relabeled below with full compound name
    )
    _draw_node_column(
        X_RIGHT, right_pos, right_species, _sp_color,
        fontsize=7, label_side="right",
    )

    # 9. Column headers
    for x, title in [
        (X_LEFT, "Producer"),
        (X_CENTER, "Compound"),
        (X_RIGHT, "Consumer"),
    ]:
        ax.text(
            x, TOTAL_H + 0.6, title,
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#111",
        )

    # 10. Flow ribbons (Bezier curves)
    # Track running y-usage per node side so successive flows stack
    left_usage = {s: left_lookup[s][0] for s in left_species}       # start at top
    center_in = {c: center_lookup[c][0] for c in center_compounds}
    center_out = {c: center_lookup[c][0] for c in center_compounds}
    right_usage = {s: right_lookup[s][0] for s in right_species}

    # Producer → compound flows
    for (sp, cid), w in sorted(producer_flows.items(), key=lambda x: -x[1]):
        if cid not in center_lookup or sp not in left_lookup:
            continue
        # Scale the ribbon thickness to the left column
        s_full, c_full = sum(left_heights), sum(center_heights)
        src_h = w / (s_full or 1.0) * sum(left_h)
        dst_h = w / (c_full or 1.0) * sum(center_h)

        y_src_top = left_usage[sp]
        y_src_bot = y_src_top - src_h
        left_usage[sp] = y_src_bot

        y_dst_top = center_in[cid]
        y_dst_bot = y_dst_top - dst_h
        center_in[cid] = y_dst_bot

        _draw_ribbon(
            ax,
            x0=X_LEFT + NODE_W / 2, y0_top=y_src_top, y0_bot=y_src_bot,
            x1=X_CENTER - NODE_W / 2, y1_top=y_dst_top, y1_bot=y_dst_bot,
            color=compound_color[cid], alpha=0.55,
        )

    # Compound → consumer flows
    for (cid, sp), w in sorted(consumer_flows.items(), key=lambda x: -x[1]):
        if cid not in center_lookup or sp not in right_lookup:
            continue
        c_full, r_full = sum(center_heights), sum(right_heights)
        src_h = w / (c_full or 1.0) * sum(center_h)
        dst_h = w / (r_full or 1.0) * sum(right_h)

        y_src_top = center_out[cid]
        y_src_bot = y_src_top - src_h
        center_out[cid] = y_src_bot

        y_dst_top = right_usage[sp]
        y_dst_bot = y_dst_top - dst_h
        right_usage[sp] = y_dst_bot

        _draw_ribbon(
            ax,
            x0=X_CENTER + NODE_W / 2, y0_top=y_src_top, y0_bot=y_src_bot,
            x1=X_RIGHT - NODE_W / 2, y1_top=y_dst_top, y1_bot=y_dst_bot,
            color=compound_color[cid], alpha=0.55,
        )

    # 11. Title
    ax.set_title(
        f"Cross-feeding Sankey  "
        f"(top {len(center_compounds)} compounds × top "
        f"{len(left_species)} producers / {len(right_species)} consumers)",
        fontsize=12, fontweight="bold", loc="left",
    )

    # 12. Draw readable compound names beside the center nodes. We skipped
    # the raw-ID text in `_draw_node_column` for the center column, so this
    # is the only label on the compound track — no overlap with the ID.
    for cid, (y_top, y_bot) in zip(center_compounds, center_pos):
        label = compound_names.get(cid, cid)[:22]
        ax.text(
            X_CENTER + NODE_W / 2 + 0.12, (y_top + y_bot) / 2,
            f"{label}\n({cid})",
            ha="left", va="center",
            fontsize=7, color="#111",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5),
        )

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    logger.info("  saved %s", out_path.name)


def _draw_ribbon(
    ax,
    x0: float, y0_top: float, y0_bot: float,
    x1: float, y1_top: float, y1_bot: float,
    color, alpha: float = 0.55,
) -> None:
    """Draw a single trapezoidal Bezier ribbon between two column nodes."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    mid = (x0 + x1) / 2
    # Top curve: (x0, y0_top) → (x1, y1_top) with cubic bezier
    verts = [
        (x0, y0_top),
        (mid, y0_top),
        (mid, y1_top),
        (x1, y1_top),
        # Right edge
        (x1, y1_bot),
        # Bottom curve back
        (mid, y1_bot),
        (mid, y0_bot),
        (x0, y0_bot),
        # Close to start
        (x0, y0_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    path = MplPath(verts, codes)
    patch = PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha, zorder=2)
    ax.add_patch(patch)


# ═══════════════════════════════════════════════════════════════════════
# Figure 5 — Category-aggregated flux
# ═══════════════════════════════════════════════════════════════════════

def _fig_categories(
    flux_raw: np.ndarray,
    compound_list: list[str],
    out_path: Path,
) -> None:
    categories = [_assign_category(c) for c in compound_list]
    cat_df = pd.DataFrame({
        "compound_id": compound_list,
        "category": categories,
        "mean_flux": flux_raw.mean(axis=0),
    })
    # Aggregate: total mean flux per category
    agg = (
        cat_df.groupby("category")["mean_flux"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={"sum": "total_flux", "count": "n_compounds"})
    )
    agg = agg.sort_values("total_flux", ascending=False)

    # Exclude "Other" for readability? Keep it.
    colors = [_PAL.get(c, _PAL["Other"]) for c in agg["category"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Total flux per category
    y1 = np.arange(len(agg))
    ax1.barh(y1, agg["total_flux"], color=colors,
             edgecolor="#333", linewidth=0.5)
    ax1.set_yticks(y1)
    ax1.set_yticklabels([
        f"{c} (n={int(n)})" for c, n in zip(agg["category"], agg["n_compounds"])
    ])
    ax1.invert_yaxis()
    ax1.set_xlabel("Summed mean flux across compounds")
    ax1.set_title(
        "(a) Category totals",
        fontsize=11, loc="left",
    )
    ax1.grid(True, axis="x", alpha=0.2)
    sns.despine(ax=ax1)

    # Panel 2: Per-compound mean flux distribution by category
    plot_df = cat_df[cat_df["mean_flux"] > 0].copy()
    order = agg["category"].tolist()
    sns.stripplot(
        data=plot_df,
        x="mean_flux", y="category",
        hue="category", palette=_PAL,
        order=order, legend=False,
        size=3.2, jitter=0.3, alpha=0.6,
        ax=ax2,
    )
    ax2.set_xlabel("Per-compound mean flux")
    ax2.set_ylabel("")
    ax2.set_title(
        "(b) Distribution within category",
        fontsize=11, loc="left",
    )
    ax2.grid(True, axis="x", alpha=0.2)
    sns.despine(ax=ax2)

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    logger.info("  saved %s", out_path.name)


# ═══════════════════════════════════════════════════════════════════════
# Per-species flux diagram:
#   upstream compounds  →  species  →  downstream compounds  →  consumer species
# ═══════════════════════════════════════════════════════════════════════

def _sanitize_filename(name: str) -> str:
    """Return a filesystem-safe filename stem for a species name."""
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch == " ":
            safe.append("_")
    return "".join(safe).strip("_") or "species"


def _fig_species_flux(
    species: str,
    abundance: np.ndarray,
    species_list: list[str],
    flux_raw: np.ndarray,
    compound_list: list[str],
    compound_names: dict[str, str],
    species_caps: dict[str, dict[str, set[str]]],
    out_path: Path,
    top_n_in: int = 6,
    top_n_out: int = 6,
    top_n_downstream: int = 5,
) -> bool:
    """Draw a 4-column flow diagram for a single species.

    Columns (left → right):
      1. **Upstream compounds** that ``species`` consumes — each ribbon
         weighted by the predicted mean flux of that compound in the
         current cohort.
      2. **The species itself**, drawn as a single large node coloured
         by genus.
      3. **Downstream compounds** that ``species`` produces — again
         weighted by predicted mean flux.
      4. **Consumer species** that take up each downstream compound,
         restricted to species actually present in the cohort (mean
         abundance > 0).

    Returns True if the figure was written, False if the species has no
    non-trivial flow to draw (no consumed *and* produced compounds).
    """
    from matplotlib.patches import Rectangle
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    caps = species_caps.get(species, {})
    p_set = caps.get("produces", set())
    c_set = caps.get("consumes", set())
    if not p_set and not c_set:
        return False

    # Intersect with the model's compound_list (1,780) so we can weight
    # each compound by predicted flux.
    compound_idx = {cid: i for i, cid in enumerate(compound_list)}
    mean_flux = flux_raw.mean(axis=0)

    def _top(cids: set[str], k: int) -> list[tuple[str, float]]:
        scored = [
            (cid, float(mean_flux[compound_idx[cid]]))
            for cid in cids if cid in compound_idx
        ]
        scored.sort(key=lambda x: -x[1])
        return [p for p in scored[:k] if p[1] > 0.0]

    in_compounds = _top(c_set, top_n_in)
    out_compounds = _top(p_set, top_n_out)
    if not in_compounds and not out_compounds:
        return False

    # Present species (for downstream consumers)
    mean_abd = abundance.mean(axis=0)
    present_species = {
        sp for sp, a in zip(species_list, mean_abd)
        if a > 0.0 and sp != species
    }

    downstream: dict[str, list[tuple[str, float]]] = {}
    for cid, _w in out_compounds:
        consumers_of_c = [
            sp for sp in present_species
            if cid in species_caps.get(sp, {}).get("consumes", set())
        ]
        # Rank by the consumer's own abundance so the most abundant
        # downstream consumers are shown.
        consumers_of_c.sort(
            key=lambda sp: float(mean_abd[species_list.index(sp)]),
            reverse=True,
        )
        downstream[cid] = [
            (
                sp,
                float(mean_abd[species_list.index(sp)]),
            )
            for sp in consumers_of_c[:top_n_downstream]
        ]

    # --- Figure geometry ------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, max(7, 0.55 * max(
        len(in_compounds), len(out_compounds), 6) + 3.5)))
    # ylim top = 11.0 leaves 1.0 for the title, 1.0 for column headers,
    # and 9.0 of vertical space for actual data rows.
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    # Column x-positions
    X_IN, X_SP, X_OUT, X_DOWN = 1.4, 5.0, 8.4, 11.0
    NODE_W = 0.28

    def _stack_y(n: int, top: float = 8.6, bot: float = 0.6) -> list[float]:
        if n == 0:
            return []
        step = (top - bot) / max(n, 1)
        return [top - step * (i + 0.5) for i in range(n)]

    in_y = _stack_y(len(in_compounds))
    out_y = _stack_y(len(out_compounds))
    # Downstream species stacked per-compound grouping: total slots
    down_items: list[tuple[str, str, float]] = []  # (compound_id, sp, abd)
    for cid, lst in downstream.items():
        for sp, abd in lst:
            down_items.append((cid, sp, abd))
    down_y = _stack_y(len(down_items) or 1)

    # --- Color maps -----------------------------------------------------
    all_compounds = [c for c, _ in in_compounds] + [c for c, _ in out_compounds]
    cmap = plt.get_cmap("tab20", max(len(all_compounds), 1))
    compound_color = {c: cmap(i) for i, c in enumerate(all_compounds)}

    # --- Draw the central species node ---------------------------------
    def _genus(name: str) -> str:
        return name.split()[0] if name else "Unknown"

    sp_rect_w = 1.2
    sp_rect_h = 2.8
    sp_center_y = 4.6
    ax.add_patch(Rectangle(
        (X_SP - sp_rect_w / 2, sp_center_y - sp_rect_h / 2),
        sp_rect_w, sp_rect_h,
        facecolor="#2e7d32", edgecolor="#111", linewidth=1.2, zorder=5,
    ))
    # Species label: dark text with white background box *below* the
    # rectangle so it is always readable regardless of the fill color.
    ax.text(
        X_SP, sp_center_y - sp_rect_h / 2 - 0.35, species,
        ha="center", va="top", fontsize=10, fontweight="bold",
        color="#111111", zorder=6,
        bbox=dict(
            facecolor="white", edgecolor="#2e7d32",
            boxstyle="round,pad=0.3", linewidth=0.9,
        ),
    )
    # Genus tag inside the green rectangle as white italic — stays
    # readable because the rectangle is solid dark green.
    ax.text(
        X_SP, sp_center_y, _genus(species),
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="white", style="italic", zorder=6,
    )

    # --- Draw columns ---------------------------------------------------
    def _node(x: float, y: float, color, label: str, side: str = "left"):
        ax.add_patch(Rectangle(
            (x - NODE_W / 2, y - 0.18), NODE_W, 0.36,
            facecolor=color, edgecolor="#333", linewidth=0.6, zorder=5,
        ))
        if side == "left":
            ax.text(
                x - NODE_W / 2 - 0.12, y, label,
                ha="right", va="center", fontsize=7.5, color="#111", zorder=6,
            )
        else:
            ax.text(
                x + NODE_W / 2 + 0.12, y, label,
                ha="left", va="center", fontsize=7.5, color="#111", zorder=6,
            )

    # Input compounds (left)
    for (cid, w), y in zip(in_compounds, in_y):
        name = (compound_names.get(cid, cid) or cid)[:26]
        _node(X_IN, y, compound_color[cid], f"{name}\n({cid})", side="left")

    # Output compounds (right of species)
    for (cid, w), y in zip(out_compounds, out_y):
        name = (compound_names.get(cid, cid) or cid)[:26]
        _node(X_OUT, y, compound_color[cid], f"{name}\n({cid})", side="right")

    # Downstream species (far right)
    for (cid, sp, abd), y in zip(down_items, down_y):
        # Use compound's own color for the marker so the link is visual
        ax.add_patch(Rectangle(
            (X_DOWN - NODE_W / 2, y - 0.18), NODE_W, 0.36,
            facecolor="#bdbdbd", edgecolor="#333", linewidth=0.5, zorder=5,
        ))
        ax.text(
            X_DOWN + NODE_W / 2 + 0.12, y, sp,
            ha="left", va="center", fontsize=7, color="#111", zorder=6,
        )

    # --- Ribbons --------------------------------------------------------
    def _ribbon(x0, y0, x1, y1, color, alpha=0.45, lw=2.0):
        mid = (x0 + x1) / 2
        verts = [
            (x0, y0),
            (mid, y0),
            (mid, y1),
            (x1, y1),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
        ]
        path = MplPath(verts, codes)
        patch = PathPatch(
            path, facecolor="none", edgecolor=color, lw=lw,
            alpha=alpha, zorder=2, capstyle="round",
        )
        ax.add_patch(patch)

    # in → species
    sp_left_x = X_SP - sp_rect_w / 2
    sp_right_x = X_SP + sp_rect_w / 2
    sp_top_y = sp_center_y + sp_rect_h / 2
    sp_bot_y = sp_center_y - sp_rect_h / 2
    max_in_w = max((w for _, w in in_compounds), default=1.0)
    for (cid, w), y in zip(in_compounds, in_y):
        # Evenly distribute the ribbon anchor along the left edge of the
        # species rectangle.
        n_in = max(len(in_compounds), 1)
        idx = in_y.index(y)
        anchor_y = sp_top_y - (idx + 1) * (sp_rect_h / (n_in + 1))
        _ribbon(
            X_IN + NODE_W / 2, y,
            sp_left_x, anchor_y,
            compound_color[cid],
            lw=0.8 + 4.0 * (w / max_in_w),
            alpha=0.55,
        )

    # species → out
    max_out_w = max((w for _, w in out_compounds), default=1.0)
    for (cid, w), y in zip(out_compounds, out_y):
        n_out = max(len(out_compounds), 1)
        idx = out_y.index(y)
        anchor_y = sp_top_y - (idx + 1) * (sp_rect_h / (n_out + 1))
        _ribbon(
            sp_right_x, anchor_y,
            X_OUT - NODE_W / 2, y,
            compound_color[cid],
            lw=0.8 + 4.0 * (w / max_out_w),
            alpha=0.55,
        )

    # out → downstream consumers
    # Group ribbon positions per compound
    out_cid_to_y = {cid: y for (cid, _), y in zip(out_compounds, out_y)}
    for (cid, sp, abd), y in zip(down_items, down_y):
        src_y = out_cid_to_y[cid]
        _ribbon(
            X_OUT + NODE_W / 2, src_y,
            X_DOWN - NODE_W / 2, y,
            compound_color[cid],
            lw=1.0,
            alpha=0.45,
        )

    # --- Column headers (row ~9.3, title above at ~10.4) ---------------
    HEADER_Y = 9.3
    ax.text(X_IN,   HEADER_Y, "Upstream\n(consumed)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(X_SP,   HEADER_Y, "Species",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(X_OUT,  HEADER_Y, "Downstream\n(produced)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(X_DOWN, HEADER_Y, "Consumers",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Title placed inside the axes, above the column headers, so it
    # never collides with the "Downstream (produced)" label.
    ax.text(
        (X_IN + X_DOWN) / 2, 10.5,
        f"Cross-feeding flow through {species}",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="#111111",
    )

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", dpi=200)
    plt.close(fig)
    logger.info("  saved flux/%s", out_path.name)
    return True


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

def generate_visualizations(
    flux_log: np.ndarray,
    flux_raw: np.ndarray,
    sample_ids: list[str],
    compound_list: list[str],
    output_dir: Path,
    top_n: int = 20,
    name_cache: Path | None = None,
    abundance: np.ndarray | None = None,
    species_list: list[str] | None = None,
    species_caps: dict[str, dict[str, set[str]]] | None = None,
    image_format: str = "png",
    flux_top_species: int = 10,
    focus_compounds: list[str] | None = None,
) -> tuple[list[Path], dict[str, str]]:
    """Generate figures into ``output_dir/images/``.

    Args:
        image_format: ``"png"`` (default — raster, good for web/slides)
            or ``"pdf"`` (vector, good for manuscripts). All figures in
            this run use the selected format.
        flux_top_species: number of most-abundant species to produce
            per-species flux diagrams for (written to ``images/flux/``).
            Set to 0 to skip that step entirely.

    Returns:
        (list of output paths, compound_id → name map for downstream reuse)
    """
    image_format = image_format.lower().lstrip(".")
    if image_format not in ("png", "pdf"):
        raise ValueError(
            f"image_format must be 'png' or 'pdf', got {image_format!r}"
        )
    ext = f".{image_format}"

    _apply_style()
    out_root = Path(output_dir)
    images_dir = out_root / "images"
    data_dir = out_root / "data"
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    def _save_tsv(name: str, df: pd.DataFrame) -> None:
        """Write a per-figure source-data TSV next to the image file.

        Every figure that the tool produces has a matching
        ``data/<name>.tsv`` so that users can re-plot the exact same
        numbers in their preferred tool without re-running inference.
        """
        path = data_dir / f"{name}.tsv"
        df.to_csv(path, sep="\t", index=False)
        logger.info("  saved data/%s.tsv", name)

    # Lookup human-readable names for ALL compounds (uses batched KEGG API +
    # on-disk cache; cheap after the first run)
    compound_names_top = fetch_compound_names(
        compound_list, cache_path=name_cache,
    )
    full_names = {c: compound_names_top.get(c, c) for c in compound_list}

    # Build the master predicted-flux DataFrame; many figures derive
    # from this.  Sorted by mean flux descending.
    means_log = flux_log.mean(axis=0)
    stds_log = flux_log.std(axis=0, ddof=0)
    means_raw = flux_raw.mean(axis=0)
    stds_raw = flux_raw.std(axis=0, ddof=0)
    cv_raw = np.where(means_raw > 0, stds_raw / np.maximum(means_raw, 1e-12), 0.0)
    df_top = pd.DataFrame({
        "compound_id":   compound_list,
        "name":          [full_names[c] for c in compound_list],
        "category":      [_assign_category(c) for c in compound_list],
        "mean_flux":     means_log,
        "mean_flux_raw": means_raw,
        "std_flux_log":  stds_log,
        "std_flux_raw":  stds_raw,
        "cv_raw":        cv_raw,
    }).sort_values("mean_flux", ascending=False)

    # focus_compounds가 설정되면 해당 compound만 필터링하여 top_n 대체
    if focus_compounds:
        focus_mask = df_top["compound_id"].isin(focus_compounds)
        if focus_mask.any():
            df_top = pd.concat([
                df_top[focus_mask],
                df_top[~focus_mask],
            ])
            top_n = focus_mask.sum()
            logger.info("  Focus compounds: %d specified, %d found", len(focus_compounds), top_n)

    outputs: list[Path] = []

    # Save the master per-compound table (top + variable figures both
    # draw from a ranked subset of this).
    _save_tsv("compound_ranking", df_top)

    # NOTE: `top_compounds` ranks by mean flux and therefore highlights
    # universal metabolites (ammonia, acetate, glutamate, …) that look
    # almost identical across samples. It is no longer part of the
    # default output — `variable_compounds` and `flux_pca` replace it
    # with sample-variation-aware alternatives. The `_fig_top_compounds`
    # helper is kept for users who want to call it from a notebook.

    # Most variable compounds across samples (CV-ranked strip plot)
    p = images_dir / f"variable_compounds{ext}"
    _fig_variable_compounds(
        flux_log=flux_log,
        flux_raw=flux_raw,
        sample_ids=sample_ids,
        compound_list=compound_list,
        compound_names=full_names,
        out_path=p,
        top_n=top_n,
    )
    outputs.append(p)
    # Dump the exact per-sample values that drive the strip plot so
    # users can recreate it in any tool.
    variable_df = df_top[df_top["std_flux_raw"] > 0].sort_values(
        "cv_raw", ascending=False,
    ).head(top_n).copy()
    for i, sid in enumerate(sample_ids):
        col_idx = [compound_list.index(cid) for cid in variable_df["compound_id"]]
        variable_df[f"log_{sid}"] = flux_log[i, col_idx]
    _save_tsv("variable_compounds", variable_df)

    # Flux-space PCA (2D sample embedding)
    p = images_dir / f"flux_pca{ext}"
    _fig_flux_pca(flux_log=flux_log, sample_ids=sample_ids, out_path=p)
    outputs.append(p)
    # Persist the PCA scores for the top 5 components so users can
    # re-plot in their preferred software.
    if len(sample_ids) >= _PCA_MIN_SAMPLES:
        X = flux_log - flux_log.mean(axis=0, keepdims=True)
        try:
            U, S, _ = np.linalg.svd(X, full_matrices=False)
            k = min(5, S.shape[0])
            scores = (U[:, :k] * S[:k])
            var_ratio = (S ** 2) / float(np.sum(S ** 2) or 1.0)
            pca_df = pd.DataFrame({
                "sample_id": sample_ids,
                "total_log_flux": flux_log.sum(axis=1),
                **{f"PC{i+1}": scores[:, i] for i in range(k)},
            })
            # Append explained variance ratios as a metadata row at
            # the top of the TSV via a companion file.
            meta = pd.DataFrame({
                "pc":  [f"PC{i+1}" for i in range(k)],
                "explained_variance_ratio": var_ratio[:k],
            })
            _save_tsv("flux_pca_scores", pca_df)
            _save_tsv("flux_pca_variance", meta)
        except np.linalg.LinAlgError:
            logger.warning("  flux_pca: SVD failed — skipping data dump")

    p = images_dir / f"flux_density{ext}"
    _fig_flux_density(flux_raw, p)
    outputs.append(p)
    # Source data: per-sample totals in both log and raw space.
    density_df = pd.DataFrame({
        "sample_id":           sample_ids,
        "total_flux_raw":      flux_raw.sum(axis=1),
        "total_flux_log":      flux_log.sum(axis=1),
        "mean_flux_raw":       flux_raw.mean(axis=1),
        "n_compounds_nonzero": (flux_raw > 0).sum(axis=1),
    })
    _save_tsv("flux_density", density_df)

    p = images_dir / f"compound_composition{ext}"
    _fig_composition(flux_raw, compound_list, full_names, sample_ids, p, top_n)
    outputs.append(p)
    # Source data: top-N compounds × all samples, long-format
    top_ids_for_comp = df_top.head(top_n)["compound_id"].tolist()
    comp_rows = []
    for i, sid in enumerate(sample_ids):
        for cid in top_ids_for_comp:
            j = compound_list.index(cid)
            comp_rows.append({
                "sample_id":   sid,
                "compound_id": cid,
                "name":        full_names.get(cid, cid),
                "flux_raw":    float(flux_raw[i, j]),
                "flux_log":    float(flux_log[i, j]),
            })
    _save_tsv("compound_composition", pd.DataFrame(comp_rows))

    p = images_dir / f"compound_categories{ext}"
    _fig_categories(flux_raw, compound_list, p)
    outputs.append(p)
    # Source data: per-category aggregated totals
    cat_df = df_top.groupby("category", as_index=False).agg(
        n_compounds=("compound_id", "size"),
        total_flux_raw=("mean_flux_raw", "sum"),
        mean_flux_raw=("mean_flux_raw", "mean"),
        total_flux_log=("mean_flux", "sum"),
        mean_flux_log=("mean_flux", "mean"),
    ).sort_values("total_flux_raw", ascending=False)
    _save_tsv("compound_categories", cat_df)

    # Structural (species_caps-based) figures. All three match the published
    # examples and require species_caps + species_list + abundance.
    if (
        abundance is not None
        and species_list is not None
        and species_caps is not None
    ):
        p_heat = images_dir / f"heatmap{ext}"
        _fig_heatmap(
            out_path=p_heat,
            species_caps=species_caps,
            top_n=20,
            abundance=abundance,
            species_list=species_list,
        )
        if p_heat.exists():
            outputs.append(p_heat)
            # Source data: top-20 × top-20 symmetric interaction matrix
            subset = _select_top_species_by_degree(
                species_caps, 20,
                abundance=abundance, species_list=species_list,
            )
            if len(subset) >= 2:
                M = _build_pair_interaction_matrix(subset, species_caps)
                heat_df = pd.DataFrame(M, index=subset, columns=subset)
                heat_df.insert(0, "species", subset)
                _save_tsv("heatmap_interaction_counts", heat_df)

        p_dist = images_dir / f"compound_distribution{ext}"
        _fig_compound_distribution(
            out_path=p_dist,
            species_caps=species_caps,
            compound_names=full_names,
            top_n=15,
            abundance=abundance,
            species_list=species_list,
        )
        if p_dist.exists():
            outputs.append(p_dist)
            # Source data: producer×consumer counts for all cross-feedable
            # compounds that have at least one producer AND one consumer.
            from collections import Counter as _Counter
            mean_abd = abundance.mean(axis=0)
            present = {
                sp for sp, a in zip(species_list, mean_abd)
                if a > 0.0 and sp in species_caps
            }
            prod_count: _Counter[str] = _Counter()
            cons_count: _Counter[str] = _Counter()
            for sp in present:
                caps = species_caps.get(sp, {})
                for cid in caps.get("produces", ()):
                    prod_count[cid] += 1
                for cid in caps.get("consumes", ()):
                    cons_count[cid] += 1
            cross = set(prod_count) & set(cons_count)
            rows = [{
                "compound_id":   cid,
                "name":          full_names.get(cid, cid),
                "category":      _assign_category(cid),
                "n_producers":   prod_count[cid],
                "n_consumers":   cons_count[cid],
                "n_interactions": prod_count[cid] * cons_count[cid],
            } for cid in cross]
            dist_df = pd.DataFrame(rows).sort_values(
                "n_interactions", ascending=False,
            )
            _save_tsv("compound_distribution", dist_df)

        p_deg = images_dir / f"crossfeeding_degree{ext}"
        _fig_crossfeeding_degree(
            out_path=p_deg,
            species_caps=species_caps,
            compound_names=full_names,
            top_n=20,
            abundance=abundance,
            species_list=species_list,
        )
        if p_deg.exists():
            outputs.append(p_deg)
            # Source data: per-species total degree + top compound
            mean_abd = abundance.mean(axis=0)
            present_list = [
                sp for sp, a in zip(species_list, mean_abd)
                if a > 0.0 and sp in species_caps
            ]
            from collections import Counter as _Counter
            p_cnt: _Counter[str] = _Counter()
            c_cnt: _Counter[str] = _Counter()
            for sp in present_list:
                caps = species_caps.get(sp, {})
                for cid in caps.get("produces", ()):
                    p_cnt[cid] += 1
                for cid in caps.get("consumes", ()):
                    c_cnt[cid] += 1
            deg_rows = []
            for sp in present_list:
                caps = species_caps.get(sp, {})
                p_set = caps.get("produces", set())
                c_set = caps.get("consumes", set())
                total = 0
                contrib: dict[str, int] = {}
                for cid in p_set:
                    k = c_cnt.get(cid, 0) - (1 if cid in c_set else 0)
                    if k > 0:
                        total += k
                        contrib[cid] = contrib.get(cid, 0) + k
                for cid in c_set:
                    k = p_cnt.get(cid, 0) - (1 if cid in p_set else 0)
                    if k > 0:
                        total += k
                        contrib[cid] = contrib.get(cid, 0) + k
                if total <= 0 or not contrib:
                    continue
                top_cid = max(contrib.items(), key=lambda x: x[1])[0]
                deg_rows.append({
                    "species":            sp,
                    "total_interactions": total,
                    "top_compound_id":    top_cid,
                    "top_compound_name":  full_names.get(top_cid, top_cid),
                })
            deg_df = pd.DataFrame(deg_rows).sort_values(
                "total_interactions", ascending=False,
            )
            _save_tsv("crossfeeding_degree", deg_df)

        # NOTE: the dense species-as-nodes / compound-as-edges "network"
        # hairball figure has been removed from the default output — it
        # was too visually noisy to interpret in practice. Per-species
        # flux diagrams (below) and the Sankey figure (also below)
        # together deliver the same information in a readable form.
        # The `_fig_species_network` helper is retained in this module
        # for users who want to call it directly from a notebook.

        p_sankey = images_dir / f"sankey{ext}"
        _fig_sankey(
            abundance=abundance,
            species_list=species_list,
            flux_raw=flux_raw,
            compound_list=compound_list,
            compound_names=full_names,
            species_caps=species_caps,
            out_path=p_sankey,
            top_n_compounds=min(top_n // 2, 10),
            top_n_producers=12,
            top_n_consumers=12,
            focus_compounds=focus_compounds,
        )
        if p_sankey.exists():
            outputs.append(p_sankey)
            # Source data: producer → compound → consumer edges for the
            # top compounds in the sankey, using the same selection
            # logic as the figure itself (rank by mean predicted flux).
            if focus_compounds:
                top_flux_idx = np.array([
                    compound_list.index(c) for c in focus_compounds
                    if c in compound_list
                ])
                n_sankey_compounds = len(top_flux_idx)
            else:
                n_sankey_compounds = min(top_n // 2, 10)
                top_flux_idx = np.argsort(-means_raw)[:n_sankey_compounds]
            mean_abd = abundance.mean(axis=0)
            present = {
                sp for sp, a in zip(species_list, mean_abd)
                if a > 0.0 and sp in species_caps
            }
            sankey_rows = []
            for ci in top_flux_idx:
                cid = compound_list[ci]
                weight = float(means_raw[ci])
                producers = [
                    sp for sp in present
                    if cid in species_caps.get(sp, {}).get("produces", set())
                ]
                consumers = [
                    sp for sp in present
                    if cid in species_caps.get(sp, {}).get("consumes", set())
                ]
                for p in producers:
                    for c in consumers:
                        if p == c:
                            continue
                        sankey_rows.append({
                            "compound_id":   cid,
                            "compound_name": full_names.get(cid, cid),
                            "producer":      p,
                            "consumer":      c,
                            "flux_weight":   weight,
                        })
            if sankey_rows:
                _save_tsv("sankey_edges", pd.DataFrame(sankey_rows))

        # Per-species flux diagrams — one figure per top-K species in
        # the cohort (K = ``flux_top_species``). Written into
        # ``images/flux/`` so the main images/ dir stays uncluttered.
        # The K-species cap keeps the output size bounded; set K = 0
        # via --flux-top-species to skip this step entirely.
        flux_written = 0
        if flux_top_species and flux_top_species > 0:
            flux_dir = images_dir / "flux"
            flux_dir.mkdir(exist_ok=True)
            mean_abd = abundance.mean(axis=0)
            mean_abd_sorted_idx = np.argsort(-mean_abd)
            top_flux_species_list: list[str] = []
            for i in mean_abd_sorted_idx:
                if mean_abd[i] <= 0:
                    break
                sp = species_list[i]
                if sp in species_caps:
                    top_flux_species_list.append(sp)
                if len(top_flux_species_list) >= flux_top_species:
                    break
        else:
            top_flux_species_list = []
            flux_dir = images_dir / "flux"  # still defined for the loop

        # Also mirror per-species data TSVs under data/flux/ when
        # per-species diagrams are enabled.
        data_flux_dir = data_dir / "flux"
        if top_flux_species_list:
            data_flux_dir.mkdir(exist_ok=True)

        mean_abd_full = abundance.mean(axis=0)
        compound_idx_lookup = {cid: i for i, cid in enumerate(compound_list)}

        for sp in top_flux_species_list:
            stem = _sanitize_filename(sp)
            p_flux = flux_dir / f"{stem}{ext}"
            ok = _fig_species_flux(
                species=sp,
                abundance=abundance,
                species_list=species_list,
                flux_raw=flux_raw,
                compound_list=compound_list,
                compound_names=full_names,
                species_caps=species_caps,
                out_path=p_flux,
                top_n_in=6,
                top_n_out=6,
                top_n_downstream=5,
            )
            if ok and p_flux.exists():
                outputs.append(p_flux)
                flux_written += 1

                # Emit a companion TSV describing the 4-column flow
                # for this species. Two-axis table: direction ∈
                # {consumed, produced, downstream_consumer}, with the
                # relevant compound / partner species / weight.
                caps = species_caps.get(sp, {})
                p_set = caps.get("produces", set())
                c_set = caps.get("consumes", set())

                def _rank(ids: set[str], k: int) -> list[tuple[str, float]]:
                    scored = [
                        (cid, float(means_raw[compound_idx_lookup[cid]]))
                        for cid in ids if cid in compound_idx_lookup
                    ]
                    scored.sort(key=lambda x: -x[1])
                    return [pq for pq in scored[:k] if pq[1] > 0.0]

                in_cpds = _rank(c_set, 6)
                out_cpds = _rank(p_set, 6)

                rows: list[dict] = []
                for cid, w in in_cpds:
                    rows.append({
                        "direction":     "consumed",
                        "species":       sp,
                        "compound_id":   cid,
                        "compound_name": full_names.get(cid, cid),
                        "partner":       "",
                        "flux_weight":   w,
                    })
                for cid, w in out_cpds:
                    rows.append({
                        "direction":     "produced",
                        "species":       sp,
                        "compound_id":   cid,
                        "compound_name": full_names.get(cid, cid),
                        "partner":       "",
                        "flux_weight":   w,
                    })
                # Downstream consumers
                for cid, w in out_cpds:
                    downstream = [
                        (sp2, float(mean_abd_full[species_list.index(sp2)]))
                        for sp2, a in zip(species_list, mean_abd_full)
                        if a > 0.0 and sp2 != sp
                        and cid in species_caps.get(sp2, {}).get("consumes", set())
                    ]
                    downstream.sort(key=lambda x: -x[1])
                    for sp2, _abd in downstream[:5]:
                        rows.append({
                            "direction":     "downstream_consumer",
                            "species":       sp,
                            "compound_id":   cid,
                            "compound_name": full_names.get(cid, cid),
                            "partner":       sp2,
                            "flux_weight":   w,
                        })
                if rows:
                    (data_flux_dir / f"{stem}.tsv").write_text(
                        pd.DataFrame(rows).to_csv(sep="\t", index=False)
                    )

        logger.info(
            "  per-species flux diagrams: %d figures in flux/",
            flux_written,
        )
    else:
        logger.info(
            "  species network + sankey + flux: skipped "
            "(abundance / species_caps not provided)"
        )

    return outputs, full_names
