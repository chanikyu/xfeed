"""xFeed: Predicting microbial cross-feeding flux from species abundance.

A goal-aligned neural flux predictor trained on 11,688 curatedMetagenomicData
samples. Given shotgun species abundance alone (no KEGG modules or
capability annotations required at inference), the model predicts
per-sample, per-compound cross-feeding flux counts for 1,780 KEGG
cross-feedable compounds derived from 4,498 KEGG REACTIONS.
"""

__version__ = "1.2.0"
