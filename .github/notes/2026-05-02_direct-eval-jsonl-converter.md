# Direct-Eval JSONL Converter — 2026-05-02

**Branch:** `exp`

## Summary

Added a dedicated converter for ProClaim direct-eval CSV outputs so SIGNOR and ConnectomeDB results can be rewritten into the shared `BaselineResult` JSONL format used by the baseline harness and analysis scripts. The change also updated the sandbox-vs-wild plotting workflow to consume the converted ProClaim JSONLs as Setting 3 by default, making the direct-eval runs comparable to the existing `single_paper` and retrieval baselines without custom CSV handling in downstream analysis.

## New Files

| File | Purpose |
|------|---------|
| `scripts/analysis/convert_direct_eval_csv_to_jsonl.py` | Converts SIGNOR and ConnectomeDB ProClaim CSV outputs into baseline-style `BaselineResult` JSONL with schema-aware claim IDs, label mapping, and repetition filtering. |

## Modified Files

| File | Changes |
|------|---------|
| `README.md` | Added the converter script to the top-level analysis usage examples. |
| `scripts/analysis/plot_sandbox_vs_wild_transitions.py` | Switched Setting 3 defaults from mixed CSV handling to converted ProClaim JSONL inputs for both SIGNOR and ConnectomeDB, and updated captions to refer to ProClaim rather than `s2_plus_ref`. |
| `.github/notes/2026-05-02_upstream-baselines-and-neurips-analysis.md` | Extended the broader May 2 note to mention the new converter and the ProClaim-backed Setting 3 defaults. |

## Architecture

```text
results/baselines/signor_direct_eval_*/results.csv
results/baselines/connectomedb_eval_*/*/results.csv
    -> scripts/analysis/convert_direct_eval_csv_to_jsonl.py
       -> schema detection: SIGNOR_ID or CDB_ID
       -> claim_id normalization
       -> label normalization
       -> BaselineResult JSONL

converted JSONL
    -> experiments.baselines.shared.evaluate.EvaluationHarness
    -> scripts/analysis/plot_sandbox_vs_wild_transitions.py
```

## Key Design Decisions

- The converter infers the dataset schema from the CSV header instead of requiring a manual `--dataset-type` flag, so the same CLI works for SIGNOR and ConnectomeDB.
- SIGNOR rows are rewritten to the baseline claim ID space `SIGNOR_ID` / `SIGNOR_ID_flip`, because that is the compatibility key used by the existing JSONL baselines and comparison scripts.
- ConnectomeDB rows keep `CDB_ID` unchanged because the dataset has no flipped-claim variants and the existing baseline JSONLs already use that identifier directly.
- The script defaults to `Repetition == 1` so the output stays one-row-per-claim like the existing baseline JSONLs, while still allowing explicit override through `--repetition`.
- SIGNOR output ordering is normalized to base claim first and flipped claim second to match the established baseline JSONL ordering and avoid spurious diffs in downstream comparisons.
- Schema-specific default metadata is filled automatically: `signor_direct_eval` / `SIGNOR` for SIGNOR and `connectomedb_direct_eval` / `ConnectomeDB` for ConnectomeDB.

## Bug Fixes

- Fixed the initial converter implementation so ConnectomeDB no longer inherited the SIGNOR-specific default `baseline_name` metadata.
- Removed the need for downstream CSV-specific parsing in the analysis workflow by converting ProClaim outputs to the same JSONL shape as the other baselines.
- Corrected Setting 3 analysis defaults so the plotting script now points to converted ProClaim JSONLs instead of older `s2_plus_ref` outputs.