# OpenScholar Verdict Parsing, Label Taxonomy & Feedback Analysis — 2026-04-08

**Branch:** `exp`

## Summary

Two main changes were made today:

1. **Verdict parsing & feedback analysis** — Deep analysis of the OpenScholar
   baseline's verdict parsing pipeline and the impact of the feedback/edit stage
   on classification accuracy across 111 SIGNOR claims.  The investigation was
   prompted by the 08_04 full-run results (accuracy=0.3784, macro_f1=0.3621)
   and determined that JSON truncation in `_parse_verdict` was not the cause of
   accuracy loss — the feedback stage's UNCERTAIN hedging was.

2. **Label taxonomy change: NEI → UNCERTAIN** — The canonical label taxonomy
   was changed from `{SUPPORT, REFUTE, NEI}` to `{SUPPORT, REFUTE, UNCERTAIN}`
   across the entire evaluation pipeline.  The previous mapping collapsed
   `UNCERTAIN → NEI`, which did not match the prompt taxonomy (which uses
   UNCERTAIN) or the claim dataset labels.  Now `UNCERTAIN` is a pass-through
   canonical label, and `NEI` maps to `UNCERTAIN` for backward compatibility.
   Metrics for both the oracle and 08_04 OpenScholar runs were recomputed.

### Verdict parsing findings

The truncated-JSON regex fallback in `_parse_verdict` works correctly — all 47
truncation cases in the 08_04 run successfully recovered the label.  The real
accuracy loss comes from the feedback/edit stage (`edit_with_feedback_retrieval`)
which causes the model to hedge decisive initial verdicts to UNCERTAIN.

A simulation comparing initial-stage vs final-output parsing showed:

| Strategy | Accuracy | Notes |
|---|---|---|
| Final output only (current) | 42/111 (0.378) | Model hedges to UNCERTAIN after feedback |
| Initial result only | 54/111 (0.487) | Pre-feedback JSON, more decisive |
| Best-of-both | 69/111 (0.622) | Use initial if non-UNCERTAIN, else fall to output |

A `claim_verdict_question_short` task prompt was also added to `instructions.py`
for future experiments with a non-JSON output format (plain label + citations),
avoiding the truncation problem entirely.

## Modified Files

| File | Changes |
|---|---|
| `experiments/baselines/shared/label_utils.py` | Canonical taxonomy changed from `{SUPPORT, REFUTE, NEI}` to `{SUPPORT, REFUTE, UNCERTAIN}`. `UNCERTAIN` is now a pass-through; `NEI` and `NOT ENOUGH INFORMATION` map to `UNCERTAIN`. Unknown labels default to `UNCERTAIN`. Duplicate `UNCERTAIN` key entries (SIGNOR and Evidence Programming) consolidated. |
| `experiments/baselines/shared/evaluate.py` | Metrics labels list changed to `["SUPPORT", "REFUTE", "UNCERTAIN"]`. Error fallback label changed from `NEI` to `UNCERTAIN`. Docstrings updated. |
| `experiments/baselines/shared/verdict.py` | Schema comments updated to reflect `SUPPORT \| REFUTE \| UNCERTAIN` taxonomy. |
| `experiments/baselines/llm_only.py` | Parse fallback default changed from `NEI` to `UNCERTAIN`. |
| `experiments/baselines/random_baseline.py` | Samples from `{SUPPORT, REFUTE, UNCERTAIN}`. Docstring updated. |
| `experiments/baselines/open_scholar_baseline.py` | Verdict fallback defaults changed from `NEI` to `UNCERTAIN`. Subprocess stdout/stderr echoing moved from `print()` to `logger.debug()`. `_parse_verdict`: drops `evidence` key, warning changed to "Could not parse JSON", added plain-text label regex. |
| `experiments/run_baselines_datasets.py` | `per_class_labels` changed to `["SUPPORT", "REFUTE", "UNCERTAIN"]`. |
| `experiments/configs/open_scholar_config.yaml` | Comment updated. |
| `experiments/configs/random_baseline_config.yaml` | Comment updated to `{SUPPORT, REFUTE, UNCERTAIN}`. |
| `OpenScholar/src/instructions.py` | Added `claim_verdict_question_short` task: plain-text output format (SUPPORT/REFUTE/UNCERTAIN + citations), no JSON. |
| `results/baselines/open_scholar/claude-sonnet-4-6_oracle/signor_metrics.json` | Recomputed with UNCERTAIN taxonomy (not tracked in git). |
| `results/baselines/open_scholar/claude-sonnet-4-6_08_04_2026/signor_metrics.json` | Recomputed with UNCERTAIN taxonomy (not tracked in git). |

## Key Design Decisions

- **UNCERTAIN as canonical third label** — The prompts, the LLM outputs, and
  the SIGNOR ground truth all use UNCERTAIN as the third category.  Mapping it
  to NEI internally created a mismatch: metrics reported a class called NEI
  that didn't appear in any prompt or dataset label.  Making UNCERTAIN the
  canonical label aligns the evaluation pipeline end-to-end.  The old `NEI`
  string is kept as an input alias in `LABEL_MAP` for backward compatibility
  with any existing JSONL result files that contain it.

- **No change to OpenScholar core pipeline** — The feedback/edit stage is part
  of the standard OpenScholar architecture.  Modifying `open_scholar.py` to
  skip feedback or change the edit prompt would alter the baseline's identity.
  The parsing layer in `open_scholar_baseline.py` is the correct place to
  handle output format issues.

- **`claim_verdict_question_short` as a separate task** — Rather than modifying
  the existing `claim_verdict_question` prompt (which produces well-structured
  JSON), a new task was added for experiments with plain-text output.  This
  avoids JSON truncation entirely since the output is just a label and optional
  citation numbers (e.g. `REFUTE [8][9]`).

- **Logger over print for subprocess output** — The previous `print(proc.stdout)`
  caused broken-pipe errors when running under `nohup`.  Using `logger.debug()`
  routes output through the logging framework, which respects file handlers.

- **Evidence field dropped from parsed JSON** — The `evidence` field in the
  model's JSON output contains citation references like `["[0]", "[1]"]`.  This
  was causing downstream validation noise without providing useful information
  for the evaluation harness.

## Analysis: Feedback Stage Causes UNCERTAIN Hedging

The `edit_with_feedback_retrieval` prompt asks the model to "incorporate the
feedback to improve the answer by including new results or details from the
retrieved passages."  When the newly retrieved S2 passages don't directly
address the specific biological claim (due to the NLP-framed keyword extraction
issue noted in 2026-04-06), the model interprets the lack of direct evidence as
reason to hedge its verdict from a decisive SUPPORT/REFUTE to UNCERTAIN.

This pattern was confirmed by examining individual logs:

1. **SIGNOR-179417** (CSK→SRC): Initial generation produces clean JSON with
   `"label": "SUPPORT"`.  After three feedback rounds with S2 retrieval, the
   edit stage preserves SUPPORT — because relevant CSK/SRC papers were found.

2. **SIGNOR-70866** (MAPK8IP2→MAPK9): Initial generation produces
   `"label": "UNCERTAIN"` (correct for this NEI claim).  The edit stage
   preserves UNCERTAIN and the final output is valid JSON.

3. **SIGNOR-256528** (GNAI1→HCK): Initial generation produces
   `"label": "REFUTE"`, but the feedback stage finds no relevant papers and the
   model hedges to UNCERTAIN.  Gold label is SUPPORT.

The feedback stage creates a systematic bias: when S2 retrieval returns
irrelevant papers (due to the NLP-domain keyword extraction), the model treats
absence of evidence as reason for uncertainty.  Combined with the
`claim_verdict_question` instruction that already prefers REFUTE over UNCERTAIN
for missing evidence, this creates a double-hedge that inflates UNCERTAIN predictions.

### Label Distribution (08_04 run)

| Label | Count | Gold Distribution |
|---|---|---|
| SUPPORT | 30 | 53 |
| REFUTE | 21 | 52 |
| UNCERTAIN | 60 | 6 |

The UNCERTAIN over-prediction (60 vs 6 gold) directly reflects the hedging effect.

### Recomputed Metrics (after NEI→UNCERTAIN taxonomy change)

| Metric | Oracle | 08_04 (adaptive retrieval) |
|---|---|---|
| Accuracy | 0.5495 | 0.3784 |
| Macro F1 | 0.4371 | 0.3621 |
| SUPPORT precision / recall | 0.84 / 0.3962 | 0.7667 / 0.434 |
| REFUTE precision / recall | 0.6842 / 0.75 | 0.7143 / 0.2885 |
| UNCERTAIN precision / recall | 0.0345 / 0.1667 | 0.0667 / 0.6667 |

## Bug Fixes

### 1. Broken pipe on subprocess stdout (fixed)
**Symptom:** `print(proc.stdout)` in `open_scholar_baseline.py` caused broken
pipe errors when the evaluation was run under `nohup` with output redirected.

**Fix:** Replaced raw `print()` calls with `logger.debug()`, which routes
through the logging framework's file handler.

### 2. Warning message accuracy (fixed)
**Symptom:** The `_parse_verdict` warning said "Truncated JSON" even when the
JSON was valid but simply couldn't be parsed for other reasons.

**Fix:** Changed to "Could not parse JSON" for accuracy.
