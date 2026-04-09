# SIGNOR Dataset Update & Baseline Completion — 2026-04-09

**Branch:** `exp`

## Summary

The SIGNOR claim verification dataset was updated (101 claims: 34 SUPPORT, 63 REFUTE, 4 UNCERTAIN; 34 flipped variants). All baseline evaluations were completed or re-run against the updated dataset. Both OpenScholar configurations (S2-retrieval and oracle-context) had 5 missing `_flip` claims each, which were filled using the resume mechanism in `run_baselines_datasets.py`. ACE, FIRE, and Random (10-repeat) baselines were also run to completion. A temporary symlink (`claude-sonnet-4-6 → claude-sonnet-4-6_08_04_2026` / `claude-sonnet-4-6_oracle`) was used to bridge the runner's output path to the existing result folders and removed after each run.

## Modified Files

| File | Change |
|------|--------|
| `results/baselines/open_scholar/claude-sonnet-4-6_08_04_2026/signor_seed100.jsonl` | Added 5 missing `_flip` claims (96 → 101 results) |
| `results/baselines/open_scholar/claude-sonnet-4-6_08_04_2026/signor_metrics.json` | Recomputed metrics on full 101-claim set |
| `results/baselines/open_scholar/claude-sonnet-4-6_oracle/signor_seed100.jsonl` | Added 5 missing `_flip` claims (96 → 101 results) |
| `results/baselines/open_scholar/claude-sonnet-4-6_oracle/signor_metrics.json` | Recomputed metrics on full 101-claim set |

## Baseline Results (SIGNOR, 101 claims)

| Baseline | Accuracy | Macro F1 | W-FPR | W-FNR | Cost (USD) |
|----------|----------|----------|-------|-------|------------|
| Random (10 repeats) | 0.3287±0.0637 | 0.2733±0.0473 | — | — | 0.00 |
| OpenScholar S2-retrieval | 0.4059 | 0.3806 | 0.1233 | 0.5941 | 4.79 |
| OpenScholar Oracle | 0.6634 | 0.5138 | 0.1103 | 0.3366 | 1.88 |
| ACE | 0.6733 | 0.5746 | — | — | 0.00 |
| FIRE | 0.7426 | 0.4874 | — | — | 0.00 |

## Key Design Decisions

- **Symlink resume pattern**: The baseline runner writes to `open_scholar/claude-sonnet-4-6/`. To resume into existing dated/oracle folders, a temporary symlink was created, then removed after completion. This avoids duplicating data or modifying the runner's output path logic.
- **Resume mechanism**: `run_baselines_datasets.py` loads completed claim IDs from the JSONL and skips them, only running missing claims. This made it efficient to fill the 5 gaps without re-running all 101 claims.
- **Missing claims were all `_flip` variants**: SIGNOR-144163_flip, SIGNOR-178679_flip, SIGNOR-179390_flip, SIGNOR-255657_flip, SIGNOR-272078_flip — these were new entries added when the dataset was updated.

## Bug Fixes

- **Dangling symlink cleanup**: After each run, the temporary `claude-sonnet-4-6` symlink was removed to avoid confusion (both result folders appeared identical when the symlink was present).
