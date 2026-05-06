# S2+Ref Baseline, Analysis Tools, and Branch Merge — 2026-05-01

**Branch:** `exp`

## Summary

Merged the `feature/ctx-manage` and `origin/motivation` branches into `exp`, integrating the S2+Ref oracle-anchor baseline, sandbox-vs-wild visualisation tooling, and section 2 motivation analysis scripts. The merge also introduced a duplicate template removal in the shared prompts module, exposed `SUFFICIENCY_BACKEND` to the evidence programming environment, and brought in notes documenting the direct-mode performance findings and single-paper baseline analysis.

## New Files

| File | Purpose |
|------|---------|
| `experiments/baselines/s2_plus_ref.py` | New `S2PlusRef` baseline: combines top-k S2 abstracts with the source paper's PubMed abstract (oracle anchor) to quantify how the reference paper biases the verdict when broader literature is also available. |
| `scripts/analysis/plot_sandbox_vs_wild_transitions.py` | Produces three-panel verdict-transition visualisation (Sankey, grouped bar, heatmap) comparing Setting 1 (single-paper) to Setting 2 (S2 top-5) across SIGNOR-Fact claims. |
| `doc/section2_motivation_analysis_plan.md` | Detailed plan for the two-claim empirical argument in paper Section 2: sandbox-is-insufficient (Analysis A) and naive-retrieval-is-insufficient (Analysis B), with proposed plots and required data sources. |
| `.github/notes/2026-04-16_single-paper-baseline-sandbox-vs-wild.md` | Backfilled note documenting the single-paper baseline implementation and its role in the Section 2 motivation analysis (added via merge). |
| `.github/notes/2026-04-22_direct-mode-performance-findings.md` | Backfilled note documenting direct-mode SIGNOR evaluation performance bottlenecks and improvement suggestions (added via merge). |

## Modified Files

| File | Changes |
|------|---------|
| `experiments/baselines/shared/prompts.py` | Removed a duplicate `VERIFICATION_USER_TEMPLATE` definition that omitted the `Output JSON.` terminator, leaving only the correct JSON-instruction version. |
| `experiments/baselines/single_paper.py` | Updated to be importable by `s2_plus_ref.py`: `_fetch_abstract` and `_format_single_passage` helpers are re-exported as module-level symbols. |
| `experiments/run_baselines_datasets.py` | Wired `s2_plus_ref` into the `build_baseline()` factory, extended CLI `--baseline` choices, and added output directory routing for the new variant. |
| `scripts/analysis/print_baseline_metrics_table.py` | Extended `BASELINE_ORDER` to include `safe` and `s2_plus_ref`; added `s2_plus_ref` to `VARIANT_ORDER` for correct sort position in the ASCII metrics table. |
| `src/proclaim/verification/config.py` | Added `SUFFICIENCY_BACKEND` to the `as_env()` mapping so that the sufficiency-classifier backend selection propagates into the Jupyter kernel environment. |
| `src/proclaim/verification/evidence_programming_direct.py` | Minor fix to expose the updated config env mapping to the direct-mode orchestrator. |
| `src/proclaim/verification/README.md` | Updated to document the verification subsystem architecture, including the direct-mode path and `SUFFICIENCY_BACKEND` config key. |

## Architecture

```text
Sandbox vs. In-the-Wild Baselines (SIGNOR-Fact, 101 claims)
============================================================

Setting 1 (in-sandbox)          Setting 2 (in-the-wild)         Setting 2+ (oracle-anchor)
─────────────────────           ────────────────────────         ──────────────────────────
PubMed efetch (PMID)            S2 top-k retrieval               PubMed efetch (PMID)
        │                               │                               │
        │ source paper abstract         │ top-5 abstracts               │ source paper abstract
        └──────────────┐               └──────────────┐               ─┘ + top-5 S2 abstracts
                       ▼                              ▼                        │
              VERIFICATION_USER_TEMPLATE     VERIFICATION_USER_TEMPLATE        │
                       │                              │                        │
                       └──────────────────────────────┘──────────────────────┘
                                       │
                                LLMBackend.complete_text()
                                       │
                                  BaselineResult JSONL
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
          plot_sandbox_vs_wild_transitions.py   print_baseline_metrics_table.py
          (Sankey + bar + heatmap)              (ASCII table, metrics log)
```

## Key Design Decisions

- `S2PlusRef` always places the source paper abstract first in the evidence list and does not deduplicate it even if S2 returns the same paper, to preserve the anchoring signal for the analysis.
- The baseline shares `_fetch_abstract` and `_format_single_passage` from `single_paper.py` to avoid duplicating PubMed retrieval and passage formatting logic.
- `s2_plus_ref` is registered as a distinct baseline name in the factory (not a config flag on `s2_retrieval`) so it generates its own output directory and can be independently targeted in SLURM batch runs.
- `SUFFICIENCY_BACKEND` is injected into the kernel environment via `as_env()` so future sufficiency classifier variants (MLP vs. LLM) can be selected without modifying kernel bootstrap code.
- The duplicate `VERIFICATION_USER_TEMPLATE` removal was triggered by the merge; the variant that omitted `Output JSON.` was causing inconsistent JSON parsing in some model runs.

## Bug Fixes

- Removed a shadowed `VERIFICATION_USER_TEMPLATE` definition in `shared/prompts.py` that was overwriting the JSON-instruction version with a plain-text version, causing downstream JSON-parse failures for baselines that imported the module after the merge.
