"""xFeed neural flux predictor.

FluxMLP learns the species → function → flux chain from shotgun species
abundance alone, without receiving KEGG module or capability annotations.
At training time it regresses against rule-based flux targets; at
inference time it predicts per-compound flux for new samples directly
from their abundance vectors.

Architecture:
    Input   (n_species,)    log1p species abundance
      ↓ LayerNorm → Linear → GELU → Dropout      × len(HIDDEN_DIMS) blocks
    Output  (n_compounds,)  softplus(log1p(flow_counts))
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    DROPOUT_RATE,
    HIDDEN_DIMS,
    N_COMPOUNDS_DEFAULT,
    N_SPECIES_DEFAULT,
)


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FluxMLPConfig:
    n_species: int = N_SPECIES_DEFAULT
    n_compounds: int = N_COMPOUNDS_DEFAULT
    hidden_dims: tuple[int, ...] = HIDDEN_DIMS
    dropout: float = DROPOUT_RATE


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════

class FluxMLP(nn.Module):
    """xFeed core: shotgun species → cross-feeding flux (log space).

    The encoder is a flat ``nn.Sequential`` of (LayerNorm, Linear, GELU,
    Dropout) layers. This matches the training-time architecture used to
    produce the released checkpoint, so its state-dict keys line up with
    the saved weights without any remapping.
    """

    def __init__(self, cfg: FluxMLPConfig | None = None):
        super().__init__()
        self.cfg = cfg or FluxMLPConfig()

        layers: list[nn.Module] = []
        prev = self.cfg.n_species
        for h in self.cfg.hidden_dims:
            layers.append(nn.LayerNorm(prev))
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.cfg.dropout))
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(prev, self.cfg.n_compounds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted log1p flux per compound (non-negative)."""
        h = self.encoder(x)
        return F.softplus(self.head(h))

    def compute_loss(
        self, pred_log_flux: torch.Tensor, target_log_flux: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(pred_log_flux, target_log_flux)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint I/O
# ═══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: FluxMLP,
    path: str | Path,
    species_list: list[str],
    compound_list: list[str],
    metrics: dict | None = None,
) -> None:
    """Persist model weights + vocabulary so predict() can reload without profiles."""
    torch.save(
        {
            "version": "1.2.0",
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": {
                "n_species": model.cfg.n_species,
                "n_compounds": model.cfg.n_compounds,
                "hidden_dims": list(model.cfg.hidden_dims),
                "dropout": model.cfg.dropout,
            },
            "species_list": list(species_list),
            "compound_list": list(compound_list),
            "metrics": metrics or {},
        },
        str(path),
    )


def load_checkpoint(
    path: str | Path, device: torch.device | None = None,
) -> tuple[FluxMLP, dict]:
    """Reload a FluxMLP checkpoint; returns (model, metadata_dict)."""
    device = device or torch.device("cpu")
    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    cfg_kwargs = ckpt.get("config", {}) or {}
    cfg = FluxMLPConfig(
        n_species=cfg_kwargs.get("n_species", N_SPECIES_DEFAULT),
        n_compounds=cfg_kwargs.get("n_compounds", N_COMPOUNDS_DEFAULT),
        hidden_dims=tuple(cfg_kwargs.get("hidden_dims", HIDDEN_DIMS)),
        dropout=cfg_kwargs.get("dropout", DROPOUT_RATE),
    )
    model = FluxMLP(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metadata = {
        "version": ckpt.get("version", "unknown"),
        "species_list": ckpt.get("species_list", []),
        "compound_list": ckpt.get("compound_list", []),
        "metrics": ckpt.get("metrics", {}),
    }
    return model, metadata
