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
RELEASE_TAG = "v1.0.0"
RELEASE_ASSET = "xfeed-data-v1.0.0.tar.gz"
RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{RELEASE_ASSET}"

# ══════════════════════════════════════════════════════════════════════
# Model architecture — FluxMLP
# ══════════════════════════════════════════════════════════════════════
# Matches the shape of the released checkpoint.
N_SPECIES_DEFAULT = 2055       # shotgun species in curatedMetagenomicData
N_COMPOUNDS_DEFAULT = 1780     # KEGG reaction-derived cross-feedable compounds
HIDDEN_DIMS = (1024, 512, 256, 128)
DROPOUT_RATE = 0.30

# ══════════════════════════════════════════════════════════════════════
# Training defaults
# ══════════════════════════════════════════════════════════════════════
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 10
DEFAULT_SEED = 42

# ══════════════════════════════════════════════════════════════════════
# Quality filter thresholds (used when xfeed train builds labels)
# ══════════════════════════════════════════════════════════════════════
MIN_ACTIVE_SPECIES = 30
MIN_KEGG_COVERAGE = 0.50
TARGET_BODY_SITE = "stool"
EXCLUDED_STUDIES = {"HMP_2019_ibdmdb"}  # reserved for cross-cohort test
