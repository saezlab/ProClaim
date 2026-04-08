# OpenScholar Verdict Parsing & Feedback Stage Analysis — 2026-04-08

**Branch:** `exp`

## Summary

Conducted a deep analysis of the OpenScholar baseline's verdict parsing pipeline
and the impact of the feedback/edit stage on classification accuracy across 111
SIGNOR claims.  The investigation was prompted by the 08_04 full-run results
(accuracy=0.3784, macro_f1=0.3621) and aimed to determine whether JSON
truncation in `_parse_verdict` was causing incorrect label assignment.

Key finding: **the truncated-JSON regex fallback in `_parse_verdict` works
correctly** — all 47 truncation cases in the 08_04 run successfully recovered
the label from the partial JSON.  The `"label"` field is always near the start
of the JSON object and is never cut off.  The real accuracy loss comes from the
feedback/edit stage (`edit_with_feedback_retrieval`) which causes the model to
hedge decisive initial verdicts to UNCERTAIN, not from parse failures.

A simulation comparing initial-stage vs final-output parsing showed:

| Strategy | Accuracy | Notes |
|---|---|---|
| Final output only (current) | 42/111 (0.378) | Model hedges to UNCERTAIN after feedback |
| Initial result only | 54/111 (0.487) | Pre-feedback JSON, more decisive |
| Best-of-both | 69/111 (0.622) | Use initial if non-NEI, else fall to output |

A `claim_verdict_question_short` task prompt was also added to `instructions.py`
for future experiments with a non-JSON output format (plain label + citations),
avoiding the truncation problem entirely.

## Modified Files

| File | Changes |
|---|---|
| `experiments/baselines/open_scholar_baseline.py` | Moved subprocess stdout/stderr echoing from raw `print()` to `logger.debug()` calls (avoids broken pipe on nohup). Per-claim log now written before stdout/stderr logging. `_parse_verdict`: drops `evidence` key from parsed JSON (validation noise). Warning message changed from "Truncated JSON" to "Could not parse JSON" for accuracy. Added plain-text label regex as first parse attempt (for `claim_verdict_question_short` format). Added fallback regex for label anywhere in text. |
| `experiments/configs/open_scholar_config.yaml` | Minor comment edit on `os_max_tokens`. |
| `OpenScholar/src/instructions.py` | Added `claim_verdict_question_short` task: plain-text output format (SUPPORT/REFUTE/UNCERTAIN + citations), no JSON. Added corresponding demonstration in `demonstrations` dict. |

## Key Design Decisions

- **No change to OpenScholar core pipeline** — The feedback/edit stage is part of the standard OpenScholar architecture. Modifying `open_scholar.py` to skip feedback or change the edit prompt would alter the baseline's identity.  The parsing layer in `open_scholar_baseline.py` is the correct place to handle output format issues.

- **`claim_verdict_question_short` as a separate task** — Rather than modifying the existing `claim_verdict_question` prompt (which produces well-structured JSON), a new task was added for experiments with plain-text output.  This avoids JSON truncation entirely since the output is just a label and optional citation numbers (e.g. `REFUTE [8][9]`).

- **Logger over print for subprocess output** — The previous `print(proc.stdout)` caused broken-pipe errors when running under `nohup`.  Using `logger.debug()` routes output through the logging framework, which respects file handlers.

- **Evidence field dropped from parsed JSON** — The `evidence` field in the model's JSON output contains citation references like `["[0]", "[1]"]`.  This was causing downstream validation noise without providing useful information for the evaluation harness.

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
   model hedges to UNCERTAIN, which normalizes to NEI.  Gold label is SUPPORT.

The feedback stage creates a systematic bias: when S2 retrieval returns
irrelevant papers (due to the NLP-domain keyword extraction), the model treats
absence of evidence as reason for uncertainty.  Combined with the
`claim_verdict_question` instruction that already prefers REFUTE over UNCERTAIN
for missing evidence, this creates a double-hedge that inflates NEI predictions.

### Label Distribution (08_04 run)

| Label | Count | Gold Distribution |
|---|---|---|
| SUPPORT | 30 | 37 (expected) |
| REFUTE | 21 | 37 (expected) |
| NEI | 60 | 37 (expected) |

The NEI over-prediction (60 vs 37 gold) directly reflects the hedging effect.

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
