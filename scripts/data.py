"""xFeed v1.1.0 data loading and on-the-fly label computation.

Three responsibilities:
  1. Loading species abundance tables (TSV / CSV / TXT)
  2. Computing per-sample cross-feeding flux labels from pre-built
     species capability profiles (no model needed — this is the v1 rule)
  3. PyTorch Dataset that pairs shotgun species abundance with log1p
     flux targets for training FluxMLP
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Abundance I/O
# ═══════════════════════════════════════════════════════════════════════

def load_abundance(path: Path | str) -> pd.DataFrame:
    """Load species abundance table (rows = samples, cols = species).

    Accepts .tsv / .txt (tab-separated) or .csv (comma-separated).
    First column is interpreted as the sample ID (index).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    sep = "," if suffix == ".csv" else "\t"
    abd = pd.read_csv(path, sep=sep, index_col=0)
    logger.info(
        "Abundance: %d samples × %d species (%s)",
        abd.shape[0], abd.shape[1], suffix,
    )
    return abd


# ═══════════════════════════════════════════════════════════════════════
# Profile loader
# ═══════════════════════════════════════════════════════════════════════

def load_profiles(
    profile_dir: Path | str,
) -> tuple[dict[str, dict[str, set[str]]], list[str]]:
    """Load (species_caps, compound_list) from a profile directory.

    Expected files in profile_dir:
      species_caps.json  — {species_name: {"produces": [...], "consumes": [...]}}
      compounds.json     — [ordered list of cross-feedable KEGG compound IDs]
    """
    profile_dir = Path(profile_dir)
    with open(profile_dir / "species_caps.json") as f:
        raw_caps = json.load(f)
    species_caps = {
        sp: {
            "produces": set(v.get("produces", [])),
            "consumes": set(v.get("consumes", [])),
        }
        for sp, v in raw_caps.items()
    }
    with open(profile_dir / "compounds.json") as f:
        compound_list = json.load(f)

    logger.info(
        "Profiles: %d species, %d cross-feedable compounds",
        len(species_caps), len(compound_list),
    )
    return species_caps, compound_list


# ═══════════════════════════════════════════════════════════════════════
# Rule-based flux label computation
# ═══════════════════════════════════════════════════════════════════════

def build_capability_matrices(
    species_list: list[str],
    species_caps: dict[str, dict[str, set[str]]],
    compound_list: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (n_species, n_compounds) boolean matrices for produces / consumes."""
    compound_idx = {c: i for i, c in enumerate(compound_list)}
    n = len(species_list)
    m = len(compound_list)
    produces_mat = np.zeros((n, m), dtype=bool)
    consumes_mat = np.zeros((n, m), dtype=bool)
    for i, sp in enumerate(species_list):
        caps = species_caps.get(sp, {})
        for c in caps.get("produces", set()):
            j = compound_idx.get(c)
            if j is not None:
                produces_mat[i, j] = True
        for c in caps.get("consumes", set()):
            j = compound_idx.get(c)
            if j is not None:
                consumes_mat[i, j] = True
    return produces_mat, consumes_mat


def compute_sample_flows(
    abundance: np.ndarray,
    produces_mat: np.ndarray,
    consumes_mat: np.ndarray,
) -> np.ndarray:
    """Compute per-sample cross-feeding flow count per compound (int32).

    For each compound c and each sample s, count unordered pairs
    (A, B) such that A produces c and B consumes c (or vice versa),
    restricted to species active in s.

    Implementation (vectorized):
        n_producers[c] = number of active species producing c
        n_consumers[c] = number of active species consuming c
        n_both[c]      = number of active species both producing AND consuming c
        flux[c]        = n_producers * n_consumers − n_both

    Returns:
        (n_samples, n_compounds) int32
    """
    n_samples = abundance.shape[0]
    n_compounds = produces_mat.shape[1]
    out = np.zeros((n_samples, n_compounds), dtype=np.int32)

    for i in range(n_samples):
        active = abundance[i] > 0
        if not active.any():
            continue
        p = produces_mat[active]
        c = consumes_mat[active]
        n_prod = p.sum(axis=0).astype(np.int64)
        n_cons = c.sum(axis=0).astype(np.int64)
        n_both = (p & c).sum(axis=0).astype(np.int64)
        out[i] = np.maximum(n_prod * n_cons - n_both, 0).astype(np.int32)

    return out


def compute_weighted_sample_flows(
    abundance: np.ndarray,
    produces_mat: np.ndarray,
    consumes_mat: np.ndarray,
    weight_fn: str = "raw",
) -> np.ndarray:
    """Compute abundance-weighted cross-feeding flux per (sample, compound).

    Instead of binary active/inactive counting, each species' contribution
    is weighted by its abundance. This captures the biological reality that
    a species at 50% abundance contributes more cross-feeding than one at
    0.01%.

    For each sample s and compound c:
        weighted_prod(c) = Σ_p  abundance(p) × produces(p, c)
        weighted_cons(c) = Σ_q  abundance(q) × consumes(q, c)
        weighted_both(c) = Σ_b  abundance(b)² × both(b, c)
        flux(s, c)       = weighted_prod × weighted_cons − weighted_both

    The self-cycling correction subtracts abundance²(b) for species that
    both produce and consume, removing the (b, b) diagonal from the
    outer product.

    Parameters
    ----------
    abundance : (n_samples, n_species) float — raw relative abundance
    produces_mat : (n_species, n_compounds) bool
    consumes_mat : (n_species, n_compounds) bool
    weight_fn : {"raw", "sqrt", "log1p"} — transform applied to abundance
        before weighting. "sqrt" dampens extreme outliers; "log1p" compresses
        further. "raw" uses relative abundance directly.

    Returns
    -------
    (n_samples, n_compounds) float32
    """
    abd = abundance.copy().astype(np.float32)

    # Apply weight transform
    if weight_fn == "sqrt":
        abd = np.sqrt(abd)
    elif weight_fn == "log1p":
        abd = np.log1p(abd * 100)  # scale to [0, ~5] range before log
    elif weight_fn != "raw":
        raise ValueError(f"Unknown weight_fn: {weight_fn!r}")

    # Vectorized: (n_samples, n_species) @ (n_species, n_compounds)
    prod_f = produces_mat.astype(np.float32)
    cons_f = consumes_mat.astype(np.float32)
    both_f = (produces_mat & consumes_mat).astype(np.float32)

    # weighted_prod[s, c] = sum_p abd[s, p] * produces[p, c]
    weighted_prod = abd @ prod_f        # (n_samples, n_compounds)
    weighted_cons = abd @ cons_f        # (n_samples, n_compounds)
    # self-cycling: sum_b abd[s, b]^2 * both[b, c]
    weighted_both = (abd ** 2) @ both_f  # (n_samples, n_compounds)

    flux = weighted_prod * weighted_cons - weighted_both
    return np.maximum(flux, 0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════

class FluxDataset(Dataset):
    """Pair shotgun species abundance with log1p cross-feeding flux labels.

    This dataset holds everything in memory. For 12k samples × 2k species
    × 2k compounds, the memory footprint is ~0.5 GB which fits comfortably
    on any modern laptop.
    """

    def __init__(
        self,
        abundance: pd.DataFrame,
        species_caps: dict[str, dict[str, set[str]]],
        compound_list: list[str],
        species_list: list[str] | None = None,
    ):
        self.compound_list = list(compound_list)
        self.species_list = (
            list(species_list) if species_list is not None
            else list(abundance.columns)
        )
        self.sample_ids = list(abundance.index)

        # Reindex abundance to the canonical species order; missing columns get 0.
        abd_aligned = abundance.reindex(columns=self.species_list).fillna(0.0)
        self.abd_raw = abd_aligned.values.astype(np.float32)
        self.features = np.log1p(self.abd_raw)

        # Compute flux labels with the v1 rule
        produces_mat, consumes_mat = build_capability_matrices(
            self.species_list, species_caps, self.compound_list,
        )
        flows = compute_sample_flows(self.abd_raw, produces_mat, consumes_mat)
        self.flow_log = np.log1p(flows.astype(np.float32))

        logger.info(
            "FluxDataset: %d samples × %d species × %d compounds  "
            "(mean flows/sample = %.0f)",
            len(self.sample_ids), len(self.species_list),
            len(self.compound_list), float(flows.sum(axis=1).mean()),
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": torch.as_tensor(self.features[idx], dtype=torch.float32),
            "flow_log": torch.as_tensor(self.flow_log[idx], dtype=torch.float32),
            "sample_id": str(self.sample_ids[idx]),
        }

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]

    @property
    def output_dim(self) -> int:
        return self.flow_log.shape[1]


def collate_fn(batch: list[dict]) -> dict:
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "flow_log": torch.stack([b["flow_log"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
    }
