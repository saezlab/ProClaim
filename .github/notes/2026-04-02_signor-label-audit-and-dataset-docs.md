# SIGNOR Label Audit & Dataset Documentation — 2026-04-02

**Branch:** `exp`

## Summary

Performed a ground-truth label audit on `results/signor_eval_results_march_30.csv` comparing it against the canonical `ground_truth.csv` from `shared_datasets/signor*/`. Found two discrepancies: (1) only 34 of the 44 `up-regulates*` edges were flipped — the 10 `WRONG` + `UNCERTAIN` edges were silently skipped; (2) `Agent_Verdict` emits `UNCERTAIN` instead of `NEI`, meaning the label normalisation step in the eval harness is required before computing metrics. Also computed accuracy using the correct label semantics. In the same session, the shared dataset documentation was pulled from `shared_datasets/claims/doc/` into `doc/`.

---

## Modified Files

| File | Changes |
|------|---------|
| `doc/claim_verification_datasets.md` | Copied from `shared_datasets/claims/doc/` — four-dataset reference (SciFact-Open, CIViC-Fact, ConnectomeDB, SIGNOR). Documents label schema, file layout, evaluation subset sizes, and `datasets/paths.yaml` configuration pattern |
| `doc/evaluation_plan.md` | Copied/updated from `shared_datasets/claims/doc/` — Group 1 (constrained retrieval) vs. Group 2 (open retrieval) evaluation design; updated SIGNOR distribution to 67 edges (34 SUPPORTED, 24 REFUTED, 9 UNCERTAIN); connectomeDB upgraded to primary benchmark role |

---

## Ground Truth Label Semantics

The CSV has two label columns. The correct ground truth for each row depends on whether the claim was flipped:

```
gold = Original_Label   if Is_Flipped == False
gold = Flipped_Label    if Is_Flipped == True
```

When `Is_Flipped=False`, `Original_Label == Flipped_Label` always (verified: no exceptions in the data). Normalization to the canonical verdict schema before metric computation:

| CSV value | Canonical |
|-----------|-----------|
| `SUPPORTED` | `SUPPORT` |
| `WRONG` | `REFUTE` |
| `UNCERTAIN` | `UNCERTAIN` |
| `Agent_Verdict: UNCERTAIN` | → `UNCERTAIN` (matches gold; no further mapping needed) |

---

## Accuracy Analysis (`signor_eval_results_march_30.csv`, n=101)

**Overall: 76.2% (77/101)**

| Subset | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Original (Is_Flipped=False) | 47 | 67 | **70.1%** |
| Flipped (Is_Flipped=True) | 30 | 34 | **88.2%** |

**Per gold-label breakdown:**

| Gold label | Correct | Total | Accuracy |
|------------|---------|-------|----------|
| SUPPORT | 27 | 34 | 79.4% |
| REFUTE | 49 | 63 | 77.8% |
| UNCERTAIN | 1 | 4 | 25.0% |

**Confusion matrix (rows = gold, cols = predicted):**

|  | REFUTE (pred) | SUPPORT (pred) | UNCERTAIN (pred) |
|--|---|---|---|
| **REFUTE (gold)** | **49** | 9 | 5 |
| **SUPPORT (gold)** | 3 | **27** | 4 |
| **UNCERTAIN (gold)** | 1 | 2 | **1** |

Key observations:
- The agent performs well on binary SUPPORT/REFUTE (79–78%), but poorly on UNCERTAIN edges (25%), tending to commit to SUPPORT or REFUTE instead.
- Flipped claims (88%) are easier than originals (70%): the flipped direction is the clearly wrong one, so the agent correctly refutes most of them.
- 9 REFUTE gold edges were predicted SUPPORT — likely the hardest errors (agent failed to detect that the SIGNOR edge is wrong).

---

## Key Design Decisions

- **Gold label selection:** use `Original_Label` when `Is_Flipped=False` and `Flipped_Label` when `Is_Flipped=True`. Since the two are always equal for non-flipped rows, `Flipped_Label` can be used as a single unified gold column as a shortcut — but the conditional formulation is the semantically correct description.
- **Label normalization is mandatory before metrics:** `Agent_Verdict` uses `UNCERTAIN`; the eval harness `normalize_label()` in `experiments/baselines/shared/label_utils.py` must be applied to both predictions and gold labels before any F1 computation.
- **Primary benchmark role assigned to ConnectomeDB:** `doc/evaluation_plan.md` now designates ConnectomeDB as the primary novel contribution benchmark (larger, cleaner labels, ligand-receptor domain); SIGNOR\* is secondary within Group 2.

---

## Bug Fixes

### Incomplete flip coverage (21 missing flipped rows)

| | Expected | Actual |
|---|---|---|
| Total rows in results CSV | 111 (67 + 44 flipped) | 101 (67 + 34 flipped) |
| Flipped up-regulates SUPPORTED | 23 | 23 ✅ |
| Flipped up-regulates WRONG | 19 | **0 ❌** |
| Flipped up-regulates UNCERTAIN | 2 | **0 ❌** |

The flip logic only ran on `SUPPORTED` edges. The 21 missing `WRONG`/`UNCERTAIN` `up-regulates*` edges need to be rerun with flipping enabled.

### Agent_Verdict vocabulary mismatch

`Agent_Verdict` emits `UNCERTAIN` (10 occurrences); the canonical verdict schema requires `NEI` in some contexts. Without `normalize_label()`, metric computation against a `NEI`-keyed schema will report zero F1 for that class. Confirmed fix: `evaluate_signor_eval_results.py` already passes verdicts through `normalize_label()` before calling the harness.

---

## Documentation Inconsistencies Found in `doc/`

Cross-referencing the docs against `shared_datasets/signor*/ground_truth.csv` (actual: **67 rows** — SUPPORTED 34, WRONG 29, UNCERTAIN 4; flippable `up-regulates*` edges: 44; expected variants: 111).

### 1. `doc/evaluation_plan.md` — Wrong SIGNOR class counts

Line: `67 edges (34 SUPPORTED, 24 REFUTED, 9 UNCERTAIN)`

| | Doc | Actual |
|---|---|---|
| SUPPORTED | 34 ✅ | 34 |
| REFUTED/WRONG | **24 ❌** | **29** |
| UNCERTAIN | **9 ❌** | **4** |
| Label name | **REFUTED ❌** | **WRONG** |

**Fix:** update to `67 edges (34 SUPPORTED, 29 WRONG, 4 UNCERTAIN)`.

### 2. `doc/baseline_implementation_plan.md` — Off-by-one edge count and wrong WRONG count

Line 289: `66 edges / **110 variants** … SUPPORTED (34), WRONG (28), UNCERTAIN (4)`

| | Doc | Actual |
|---|---|---|
| Total edges | **66 ❌** | **67** |
| WRONG | **28 ❌** | **29** |
| Expected variants | **110 ❌** | **111** (67 + 44) |

**Fix:** `67 edges / **111 variants** … SUPPORTED (34), WRONG (29), UNCERTAIN (4)`.  
Same error appears in the Fairness Checklist (line ~324): `# 110 variants total: 66 forward + 44 flipped` → should be `# 111 variants total: 67 forward + 44 flipped`.

### 3. `doc/claim_verification_datasets.md` vs `doc/baseline_implementation_plan.md` — Canonical label for uncertainty disagrees

| Doc | Canonical label for UNCERTAIN |
|-----|------------------------------|
| `claim_verification_datasets.md` line 9 | `NEI / UNCERTAIN → UNCERTAIN` (canonical = **UNCERTAIN**) |
| `baseline_implementation_plan.md` line 80 | `UNCERTAIN → NEI` (canonical = **NEI**) |
| Summary table in `claim_verification_datasets.md` line 257 | uses **UNCERTAIN** as canonical |

The two docs use different canonical vocabularies (`UNCERTAIN` vs `NEI`) for the same class. The evidence programming system itself emits `UNCERTAIN` — pick one and apply consistently. **Recommendation:** use `UNCERTAIN` as canonical (matches the system output, avoids confusion with FEVER-style NEI), and update `baseline_implementation_plan.md` accordingly.

### 4. `doc/claim_verification_datasets.md` — Internal inconsistency on down-regulates flip

Lines 208–209 state: *"Negative (down-regulates variants) with flip=True → claim becomes activation"*  
Line 212 immediately contradicts: *"`build_datasets.py` only calls `flip=True` for `EFFECT ∈ {up-regulates, …}`. Down-regulates … are left as-is."*

**Fix:** remove lines 208–209 (or rewrite them to say down-regulates are **not** flipped).

### 5. `doc/claim_verification_datasets.md` — Wrong total variant count for SIGNOR

Line 258: `All 66 forward + 45 negated (up-regulates only) | 111 variants`  
66 + 45 = 111, but the correct breakdown is **67 forward + 44 flipped = 111**.

**Fix:** `All 67 forward + 44 negated (up-regulates only) | 111 variants`.
