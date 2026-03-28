# Scientific Claim Verification Datasets

Four datasets under `/hps/nobackup/saezrodriguez/shared_datasets/` used for evaluating the PKEvolve verification pipeline.

**Label mapping across datasets:** SUPPORT / SUPPORTS → `Yes`; CONTRADICT / REFUTES / WRONG → `No`; NEI / UNCERTAIN → `None`.

---

## 1. SciFact-Open

**Path:** `/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/`

| File | Description |
|------|-------------|
| `data/claims.jsonl` | 279 claims; 206 have evidence annotations, 73 are NEI |
| `data/corpus.jsonl` | 500 K-abstract retrieval corpus |

**Claim schema:**
```jsonc
{
  "id": number,
  "claim": string,
  "evidence": {           // empty dict if NEI
    "<doc_id>": {
      "label": "SUPPORT" | "CONTRADICT",
      "sentences": [number]  // sentence indices in corpus abstract
    }
  }
}
```

### Evaluation subset

Use the **20 % held-out test split** of the 206 annotated claims (those with `evidence != {}`), stratified by consensus label, matching the sufficiency classifier's own `random_state=42`. This produces ~41 claims not used in classifier pool construction.

```python
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

claims_path = Path("/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/claims.jsonl")
with open(claims_path) as f:
    all_claims = [json.loads(l) for l in f if l.strip()]

annotated = [c for c in all_claims if c.get("evidence")]

def consensus_label(claim):
    labels = [info["label"] for info in claim["evidence"].values()]
    return "SUPPORT" if all(l == "SUPPORT" for l in labels) else "CONTRADICT"

labels = [consensus_label(c) for c in annotated]
_, test_claims = train_test_split(annotated, test_size=0.2, stratify=labels, random_state=42)
# ~41 claims: ~20 SUPPORT, ~21 CONTRADICT
```

**Claim string:** use `claim["claim"]` directly. Evidence text comes from `corpus.jsonl` via `doc_id`, selecting sentences at `doc["abstract"][i]` for `i` in `evidence[doc_id]["sentences"]`.

---

## 2. CIViC-Fact

**Path:** `/hps/nobackup/saezrodriguez/shared_datasets/civicfact/data_builder/builds/civicfact-2025.03.25/data.jsonl.gz`

| Partition | Rows | SUPPORTS | REFUTES | NEI |
|-----------|------|----------|---------|-----|
| train | 6 428 | 2 162 | 2 091 | 2 175 |
| dev | 2 099 | 712 | 676 | 711 |
| **test** | **2 055** | **689** | **666** | **700** |
| unassigned | 12 657 | — | — | — |

**Key fields:** `claim.flat` (natural-language claim string), `evidence.flat` (concatenated evidence text), `gold_label_name` (`SUPPORTS` / `REFUTES` / `NEI`), `partition`, `document.pmid`, `flagged` (exclude if `True`).

### Evaluation subset

Primary evaluation uses `partition == "test"` (2,055 rows). `train` + `dev` reserved for fine-tuning or few-shot sampling.

```python
import gzip, json
from pathlib import Path

civic_path = Path(
    "/hps/nobackup/saezrodriguez/shared_datasets/civicfact"
    "/data_builder/builds/civicfact-2025.03.25/data.jsonl.gz"
)
with gzip.open(civic_path, "rt") as f:
    rows = [json.loads(l) for l in f if l.strip()]

test_rows = [r for r in rows if r.get("partition") == "test" and not r.get("flagged")]
# ~2 055 rows after deduplication; drop flagged=True entries
```

**Claim string:** `row["claim.flat"]`. Evidence: `row["evidence.flat"]`.

---

## 3. ConnectomeDB

**Path:** `/hps/nobackup/saezrodriguez/shared_datasets/connectomedb/`

### Evaluation subset

| File | Rows | Role |
|------|------|------|
| `cdb25_direct_multipub_unique.csv` | 184 | Positives — CDB25 Direct pairs, ≥2 publications, deduplicated |
| `ConnectomeDB2020_rejected_labeled.csv` | 363 | Negatives/NEI — CDB2020 pairs rejected by CDB25 curators |

**Key positive columns:** `LR Pair` (e.g. `TGFB1 TGFBR1`), `Ligand Symbols`, `Receptor Symbols`, `AI summary` (Perplexity URL embedding PMIDs), `Species`.

**Key negative columns:** `LR_pair`, `Ligand`, `Receptor`, `Label` (`REFUTED` / `NEI`), `Rejection_reason`, `Curator comments`.

```python
import pandas as pd
from pathlib import Path

base = Path("/hps/nobackup/saezrodriguez/shared_datasets/connectomedb")
df_pos = pd.read_csv(base / "cdb25_direct_multipub_unique.csv")       # 184 rows, label=SUPPORTED
df_neg = pd.read_csv(base / "ConnectomeDB2020_rejected_labeled.csv")   # 363 rows, label=REFUTED|NEI
```

**Claim string:** form from `LR Pair` columns, e.g. `"{Ligand} directly binds to and activates {Receptor} as a ligand–receptor pair."` for positives; equivalently negated for REFUTED entries.

---

## 4. SIGNOR Ground Truth

**Path:** `/hps/nobackup/saezrodriguez/shared_datasets/signor*/ground_truth.csv`

| Metric | Value |
|--------|-------|
| Total edges | 66 |
| SUPPORTED | 34 (51.5 %) |
| UNCERTAIN | 4 (6.1 %) |
| WRONG | 28 (42.4 %) |

**Key columns:** `SIGNOR_ID`, `ENTITYA`, `ENTITYB`, `EFFECT`, `Label` (`SUPPORTED` / `UNCERTAIN` / `WRONG`), `SENTENCE` (gold evidence sentence), `PMID`, `MECHANISM`, `DIRECT`.

**Effect distribution (up-regulates variants = flippable):**

| Effect | Count | Flippable |
|--------|-------|-----------|
| up-regulates | 22 | yes |
| up-regulates activity | 17 | yes |
| down-regulates activity | 10 | no |
| down-regulates | 9 | no |
| up-regulates quantity | 3 | yes |
| up-regulates quantity by expression | 2 | yes |
| down-regulates quantity by destabilization | 2 | no |
| unknown | 2 | no |

### Evaluation subset

All 66 forward claims **plus** negated variants for the 44 up-regulates edges — **110 claim variants** per repetition. Flip logic: only `EFFECT ∈ {up-regulates, up-regulates activity, up-regulates quantity, up-regulates quantity by expression}` is flipped (activation → inhibition). Down-regulates and non-directional effects are left as-is.

| Variant | Expected label |
|---------|---------------|
| `flip=False` | Original `Label` |
| `flip=True` (up-regulates only) | SUPPORTED → WRONG; WRONG → SUPPORTED; UNCERTAIN → UNCERTAIN |

**Claim construction:** always use `construct_signor_claim()` from `experiments/run_signor_eval.py`; do not construct strings manually.

```python
import glob, pandas as pd
from experiments.run_signor_eval import construct_signor_claim, get_flipped_label

gt_path = glob.glob("/hps/nobackup/saezrodriguez/shared_datasets/signor*/ground_truth.csv")[0]
df = pd.read_csv(gt_path)

claims = []
for _, row in df.iterrows():
    for flip in [False, True]:
        claim_str = construct_signor_claim(row["ENTITYA"], row["ENTITYB"], row["EFFECT"], flip=flip)
        if claim_str is None:   # non-flippable effects return same as flip=False; deduplicate
            continue
        label = get_flipped_label(row["Label"], flip=flip)
        claims.append({"id": row["SIGNOR_ID"], "flip": flip, "claim": claim_str, "label": label})
```

---

## Summary

| Dataset | Evaluation subset | Size | Labels |
|---------|-------------------|------|--------|
| SciFact-Open | 20 % stratified test split of annotated claims | ~41 claims | SUPPORT / CONTRADICT |
| CIViC-Fact | `partition == "test"`, `flagged != True` | ~2 055 rows | SUPPORTS / REFUTES / NEI |
| ConnectomeDB | `cdb25_direct_multipub_unique.csv` + `ConnectomeDB2020_rejected_labeled.csv` | 184 + 363 rows | SUPPORTED / REFUTED / NEI |
| SIGNOR | All 66 forward + 44 negated (up-regulates only) | 110 variants | SUPPORTED / WRONG / UNCERTAIN |
