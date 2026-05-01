# ConnectomeDB Reference-Paper Baselines — 2026-05-01

**Branch:** `exp`

## Summary

Prepared the ConnectomeDB claim dataset for the reference-paper baselines by ensuring every claim row carries a usable source PMID, including all 184 SUPPORT rows. This makes the existing `single_paper` and `s2_plus_ref` baselines runnable on ConnectomeDB without falling back to `UNCERTAIN` for positive examples. The baseline runner was also aligned so `s2_plus_ref` uses the same shared LLM builder and CLI arguments as the other LLM-backed baselines.

## Modified Files

| File | Changes |
|------|---------|
| `/hps/nobackup/saezrodriguez/shared_datasets/claims/datasets/connectomedb.csv` | Filled the `pmid` column for supported ConnectomeDB claims using the first paper in the evidence URL, bringing PMID coverage to all 318 rows. |
| `experiments/run_baselines_datasets.py` | Routed `single_paper`, `retrieval`, and `s2_plus_ref` through the shared `build_llm_backend()` path and ensured `s2_plus_ref` respects `--model`, `--max-tokens`, `--reasoning-effort`, `--thinking-budget`, `--top-k`, and `--strip-query`. |
| `experiments/baselines/single_paper.py` | Existing single-paper baseline now applies cleanly to ConnectomeDB because the required `pmid` field is present for all claims. |
| `experiments/baselines/s2_plus_ref.py` | Existing reference-plus-retrieval baseline can now anchor ConnectomeDB runs on the source abstract before appending retrieved S2 evidence. |

## Architecture

```text
ConnectomeDB reference-aware baselines
======================================

connectomedb.csv
  claim, label, pmid
        |
        +-------------------------------+
        |                               |
        v                               v
single_paper                      s2_plus_ref
PubMed efetch(pmid)              PubMed efetch(pmid)
1 source abstract                + Semantic Scholar top-k
        |                               |
        +---------------+---------------+
                        |
                        v
              VERIFICATION_USER_TEMPLATE
                        |
                        v
                 shared LLM backend
                        |
                        v
               BaselineResult JSONL
```

## Key Design Decisions

- Use the first cited paper as the reference document for ConnectomeDB SUPPORT claims. This matches the current dataset update and keeps the source-paper condition deterministic.
- Keep the baseline code dataset-agnostic rather than adding ConnectomeDB-specific branches to `single_paper` or `s2_plus_ref`. Once `pmid` is normalized in the dataset, the existing baseline interfaces are sufficient.
- Reuse `build_llm_backend()` for `s2_plus_ref` so model selection, token limits, and reasoning controls stay consistent across all LLM-backed baselines.
- Keep query stripping controlled by the existing shared `--strip-query` flag so ConnectomeDB retrieval behavior stays aligned with the current runner surface.

## Bug Fixes

- Fixed `s2_plus_ref` runner wiring in `experiments/run_baselines_datasets.py`. The baseline was reading stale argument names (`s2_model`, `s2_top_k`, `s2_strip_parens`) instead of the active shared CLI options, which could make runs ignore the requested model, retrieval depth, or query preprocessing.
