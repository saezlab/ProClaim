"""
Configuration for τ-threshold ablation study.
"""

# The complete list of τ thresholds to evaluate in the ablation study.
# 0.00: baseline - all conflicts are kept as negative_conflict regardless of imbalance
# 0.50: most aggressive relabeling - only perfectly balanced 50/50 are kept as conflict
# 1.00: relabel ALL negative_conflict samples (r_minority max = 0.5, so always < 1.0)
TAU_THRESHOLDS = [
    0.00, 0.11, 0.13, 0.14, 0.17, 0.20, 
    0.25, 0.29, 0.33, 0.38, 0.40, 0.43, 0.50,
    1.00
]
