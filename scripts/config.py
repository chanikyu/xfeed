"""xFeed configuration — KEGG API, defaults, release URLs."""
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# KEGG API
# ══════════════════════════════════════════════════════════════════════
KEGG_API = "https://rest.kegg.jp"
REQUEST_DELAY = 0.35           # KEGG API rate limit (seconds between calls)
KEGG_BATCH_SIZE = 10           # max IDs per /get batch

# ══════════════════════════════════════════════════════════════════════
# User-facing defaults
# ══════════════════════════════════════════════════════════════════════
DEFAULT_DATA_DIR = Path.home() / ".xfeed"

# GitHub Releases — pre-built profiles + model weights
GITHUB_REPO = "chanikyu/xfeed"
RELEASE_TAG = "v1.2.0"
RELEASE_ASSET = "xfeed-data-v1.2.0.tar.gz"
RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{RELEASE_ASSET}"

# ══════════════════════════════════════════════════════════════════════
# Model architecture — FluxMLP (v1.2.0 canonical)
# ══════════════════════════════════════════════════════════════════════
# Same MLP-small architecture as v1.1.0 (~497 K parameters).
# v1.2.0 adds concordance-aware multi-task training:
#   Loss = MSE + β×concordance_hinge + γ×metabolomics_correlation
# This improves literature concordance from 46% to 92% (12/13 diseases)
# and HMP2 metabolomics correlation (butyrate ρ 0.04 → 0.51).
# Legacy checkpoints archived under paper/tool/model/legacy/.
N_SPECIES_DEFAULT = 2055       # shotgun species in curatedMetagenomicData
N_COMPOUNDS_DEFAULT = 1780     # KEGG reaction-derived cross-feedable compounds
HIDDEN_DIMS = (128,)           # v1.2.0: one 128-unit hidden layer (same as v1.1.0)
DROPOUT_RATE = 0.30

# ══════════════════════════════════════════════════════════════════════
# Label generation — Stage 1 upgrade: abundance-weighted labels
# ══════════════════════════════════════════════════════════════════════
# "binary": original v1.0/v1.1 pair counting (active/inactive species)
# "weighted": Stage 1 abundance-weighted (flux ∝ abundance(p) × abundance(q))
LABEL_TYPE = "weighted"
WEIGHT_FN = "raw"  # "raw" | "sqrt" | "log1p" — transform on abundance before weighting

# ══════════════════════════════════════════════════════════════════════
# Training defaults
# ══════════════════════════════════════════════════════════════════════
DEFAULT_EPOCHS = 60
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_PATIENCE = 10
DEFAULT_SEED = 42

# ══════════════════════════════════════════════════════════════════════
# Quality filter thresholds (used when xfeed train builds labels)
# ══════════════════════════════════════════════════════════════════════
MIN_ACTIVE_SPECIES = 30
MIN_KEGG_COVERAGE = 0.50
TARGET_BODY_SITE = "stool"
EXCLUDED_STUDIES = {"HMP_2019_ibdmdb"}  # reserved for cross-cohort test
