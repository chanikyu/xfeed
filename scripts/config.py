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
RELEASE_TAG = "v1.1.0"
RELEASE_ASSET = "xfeed-data-v1.1.0.tar.gz"
RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{RELEASE_ASSET}"

# ══════════════════════════════════════════════════════════════════════
# Model architecture — FluxMLP (v1.1.0 canonical)
# ══════════════════════════════════════════════════════════════════════
# Matches the shape of the released checkpoint. v1.1.0 promotes the
# single-hidden-layer MLP-small (~497 K parameters) to canonical after
# the §2.9 ablation showed it outperforms the 3.03 M four-layer v1.0.0
# on every headline metric (see paper/manuscript/draft_v4.txt).
# The legacy four-layer checkpoint is archived under
# paper/tool/model/legacy/xfeed_v1_0_0.pt for reproducibility.
N_SPECIES_DEFAULT = 2055       # shotgun species in curatedMetagenomicData
N_COMPOUNDS_DEFAULT = 1780     # KEGG reaction-derived cross-feedable compounds
HIDDEN_DIMS = (128,)           # v1.1.0: one 128-unit hidden layer
DROPOUT_RATE = 0.30

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
