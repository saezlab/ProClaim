# Scientific Claim Verification Datasets

Four datasets used for evaluating the PKEvolve verification pipeline.

**Primary datasets (open retrieval, claim-level verdict):** ConnectomeDB, SIGNOR — these are the novel benchmark contribution; no prior dataset provides claim-level labels for open-retrieval scientific verification in these domains.

**Secondary datasets (constrained retrieval, diagnostic):** SciFact-Open, CIViC-Fact — included for task decomposition (isolating retrieval from reasoning failures) and community comparability with prior work.

**Label mapping across datasets:** SUPPORTED / SUPPORTS → `SUPPORT`; CONTRADICT / REFUTES / WRONG → `REFUTE`; NEI / UNCERTAIN → `UNCERTAIN`.

**Configuration:** All source paths are stored in `datasets/paths.yaml` (git-ignored, not committed). Copy `datasets/paths.yaml.template` to `datasets/paths.yaml` and fill in the paths for your environment before running any code below.

---

## 1. SciFact-Open *(secondary — constrained retrieval)*

**Path:** `sources.scifact_dir` in `datasets/paths.yaml`

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
import yaml
from sklearn.model_selection import train_test_split

cfg = yaml.safe_load(open(Path(__file__).parent.parent / "datasets/paths.yaml"))
scifact_dir = Path(cfg["sources"]["scifact_dir"])

claims_path = scifact_dir / "claims.jsonl"
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

**Paper references in built CSV:** `paper_ids` (semicolon-separated Semantic Scholar `doc_id` values for all evidence documents) and `paper_titles` (corresponding semicolon-separated paper titles from `corpus.jsonl`).

---

## 2. CIViC-Fact *(secondary — constrained retrieval)*

**Path:** `sources.civic_path` in `datasets/paths.yaml`

| Partition | Rows | SUPPORTS | REFUTES | NEI |
|-----------|------|----------|---------|-----|
| train | 6 428 | 2 162 | 2 091 | 2 175 |
| dev | 2 099 | 712 | 676 | 711 |
| **test** | **2 055** | **689** | **666** | **700** |
| unassigned | 12 657 | — | — | — |

**Key fields:** `claim.flat` (natural-language claim string), `evidence.flat` (concatenated evidence text), `gold_label_name` (`SUPPORTS` / `REFUTES` / `NEI`), `partition`, `document.pmid`, `document.pmcid`, `document.doi`, `document.title`, `flagged` (exclude if `True`). Note: these are flat top-level keys in the JSONL record, not nested under a `document` object.

**Paper references in built CSV:** `pmid`, `pmcid`, `doi`, `paper_title` (all sourced from the `document.*` keys above; `id` column is also set to the PMID).

### Evaluation subset

Primary evaluation uses `partition == "test"` (2,055 rows). `train` + `dev` reserved for fine-tuning or few-shot sampling.

```python
import gzip, json
from pathlib import Path
import yaml

cfg = yaml.safe_load(open(Path(__file__).parent.parent / "datasets/paths.yaml"))
civic_path = Path(cfg["sources"]["civic_path"])

with gzip.open(civic_path, "rt") as f:
    rows = [json.loads(l) for l in f if l.strip()]

test_rows = [r for r in rows if r.get("partition") == "test" and not r.get("flagged")]
# ~2 055 rows after deduplication; drop flagged=True entries
```

**Claim string:** `row["claim.flat"]`. Evidence: `row["evidence.flat"]`.

### Natural-language claim templates

`claim.flat` stores a tab-separated table (not a sentence). When converting to natural-language form, parse the structured `claim` field and apply the templates below based on `(evidenceType, significance)`. The `{variant}` placeholder is the `molecularProfile` value (e.g. `EGFR T790M`, `KRAS G12V`, `BCR::ABL1 Fusion`) and is used verbatim — no "mutation" prefix — to accommodate fusions, overexpression, and copy-number events.

| evidenceType | significance | Count (test) | Natural-language template |
|---|---|---|---|
| PREDICTIVE | sensitivity/response | 632 | `{variant} is associated with sensitivity to {therapy} in {disease}.` |
| PREDICTIVE | resistance | 601 | `{variant} confers resistance to {therapy} in {disease}.` |
| PREDICTIVE | reduced sensitivity | 5 | `{variant} is associated with reduced sensitivity to {therapy} in {disease}.` |
| PREDICTIVE | adverse response | 4 | `{variant} is associated with adverse response to {therapy} in {disease}.` |
| PROGNOSTIC | poor outcome | 180 | `{variant} is associated with poor prognosis in patients with {disease}.` |
| PROGNOSTIC | better outcome | 122 | `{variant} is associated with better prognosis in patients with {disease}.` |
| PREDISPOSING | predisposition | 115 | `{variant} predisposes to {disease}.` (append `, presenting as {phenotype}` when phenotype present) |
| PREDISPOSING | uncertain significance | 15 | `{variant} is a variant of uncertain significance with respect to predisposition to {disease}.` |
| PREDISPOSING | na | 14 | `{variant} has no established disease predisposition.` |
| DIAGNOSTIC | positive | 103 | `{variant} is a positive diagnostic marker for {disease}.` (append `, associated with {phenotype}` when phenotype present) |
| DIAGNOSTIC | negative | 1 | `{variant} is not a diagnostic marker for {disease}.` |
| FUNCTIONAL | loss of function | 59 | `{variant} results in loss of function.` (append ` in {disease}` when disease present) |
| FUNCTIONAL | gain of function | 58 | `{variant} results in gain of function.` (append ` in {disease}` when disease present) |
| FUNCTIONAL | unaltered function | 56 | `{variant} does not alter protein function.` |
| FUNCTIONAL | dominant negative | 14 | `{variant} acts as a dominant negative.` |
| FUNCTIONAL | neomorphic | 5 | `{variant} is neomorphic, conferring an altered or new molecular function.` |
| ONCOGENIC | oncogenicity | 30 | `{variant} has oncogenic activity in {disease}.` |

**Optional fields:** `therapies`, `diseases`, and `phenotypes` may each contain multiple values; join with ` or ` for therapies and ` / ` for diseases and phenotypes.

**Examples:**

```
NT5C2 K359Q confers resistance to Arabinosylguanine or Nelarabine in T-cell Acute Lymphoblastic Leukemia.
EGFR T790M is associated with poor prognosis in patients with Lung Non-small Cell Carcinoma.
CTCF P378L results in loss of function.
NF2 c.1396C>T predisposes to Adult Spinal Cord Ependymoma.
KRAS Q61H has oncogenic activity in Cancer.
```

---

## 3. ConnectomeDB *(primary — open retrieval)*

**Path:** `sources.connectome_dir` in `datasets/paths.yaml`

### Evaluation subset

Two source files from `connectome_dir` are merged into the consolidated `connectomedb.csv`:

| File | Rows | Labels |
|------|------|--------|
| `cdb25_direct_multipub_unique.csv` | 184 | SUPPORT (all 184) |
| `ConnectomeDB2020_rejected_labeled.csv` | 134 | REFUTE: 121 · UNCERTAIN: 13 |
| **Total** | **318** | **SUPPORT: 184 · REFUTE: 121 · UNCERTAIN: 13** |

The negative set excludes 228 "No primary evidence" pairs (may have indirect support) and 1 species-specific artefact. Species breakdown (positives only): human 145, mouse 36, rat 3.

**Key positive columns:** `LR Pair` (e.g. `TGFB1 TGFBR1`), `Ligand Symbols`, `Receptor Symbols`, `AI summary` (Perplexity URL embedding PMIDs), `Species`.

**Key negative columns:** `LR_pair`, `Ligand`, `Receptor`, `Label` (`REFUTED` / `NEI`), `Rejection_reason`, `Curator comments`, `PMID` (PubMed ID of the original supporting publication, e.g. `PMID:12943195`). No paper IDs are available for the positive set (source has only a Perplexity search URL in `AI summary`).

```python
import pandas as pd
from pathlib import Path
import yaml

cfg = yaml.safe_load(open(Path(__file__).parent.parent / "datasets/paths.yaml"))
output_dir = Path(cfg["output_dir"])

# Load the pre-built consolidated dataset
df = pd.read_csv(output_dir / "connectomedb.csv")
# df['label'] values: SUPPORT (184), REFUTE (121), UNCERTAIN (13)
```

**Claim string:** all entries use the affirmative template `"In the context of protein-protein interactions, {Ligand} as ligand directly interacts with {Receptor} as receptor."`. The `label` column carries the verdict (SUPPORT / REFUTE / UNCERTAIN).

---

## 4. SIGNOR Ground Truth *(primary — open retrieval)*

**Path:** `sources.signor_path` in `datasets/paths.yaml`

| Metric | Value |
|--------|-------|
| Total edges | 67 |
| SUPPORTED | 34 (50.7 %) |
| UNCERTAIN | 4 (6.0 %) |
| WRONG | 29 (43.3 %) |

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

All 67 forward claims **plus** negated variants for the 44 up-regulates edges — **111 claim variants** per repetition.

Flip logic in `construct_signor_claim()`:
- `up-regulates` variants with `flip=True` → claim becomes inhibition; label inverted (SUPPORTED↔WRONG, UNCERTAIN unchanged).

`build_datasets.py` only calls `flip=True` for `EFFECT ∈ {up-regulates, up-regulates activity, up-regulates quantity, up-regulates quantity by expression}`. Down-regulates and non-directional effects are **not** flipped — they are left as-is in the dataset.

| Variant | Expected label |
|---------|---------------|
| `flip=False` | Original `Label` |
| `flip=True` (up-regulates only) | SUPPORT → REFUTE; REFUTE → SUPPORT; UNCERTAIN → UNCERTAIN |

**Claim templates** (produced by `construct_signor_claim()` in `build_datasets.py`):
- Activation: `"{source} directly activates {target} (either through post-translational modification, complex formation, or direct regulation of expression)."`
- Inhibition: `"{source} directly inhibits {target} (either through post-translational modification, complex formation, or direct regulation of expression)."`
- Binding: `"{source} directly interacts with {target} (e.g., physical binding)."`

**Claim construction:** always use `construct_signor_claim()` from `build_datasets.py`; do not construct strings manually.

```python
import glob
import pandas as pd
import yaml
from pathlib import Path
from build_datasets import construct_signor_claim, get_flipped_label

cfg = yaml.safe_load(open(Path(__file__).parent.parent / "datasets/paths.yaml"))
gt_path = glob.glob(cfg["sources"]["signor_path"])[0]
df = pd.read_csv(gt_path)

claims = []
for _, row in df.iterrows():
    for flip in [False, True]:
        # Skip flip=True for non-flippable effects (would produce duplicate claim)
        if flip and row["EFFECT"] not in {
            "up-regulates", "up-regulates activity",
            "up-regulates quantity", "up-regulates quantity by expression",
        }:
            continue
        claim_str = construct_signor_claim(row["ENTITYA"], row["ENTITYB"], row["EFFECT"], flip=flip)
        label = get_flipped_label(row["Label"], flip=flip)
        claims.append({"id": row["SIGNOR_ID"], "flip": flip, "claim": claim_str, "label": label})
```

---

## Summary

| Dataset | Role | Evaluation subset | Size | Labels |
|---------|------|-------------------|------|--------|
| **ConnectomeDB** | **Primary** | `cdb25_direct_multipub_unique.csv` + `ConnectomeDB2020_rejected_labeled.csv` → `connectomedb.csv` | 318 rows (SUPPORT: 184, REFUTE: 121, UNCERTAIN: 13) | SUPPORT / REFUTE / UNCERTAIN |
| **SIGNOR** | **Primary** | All 67 forward + 44 negated (up-regulates only) | 111 variants | SUPPORT / REFUTE / UNCERTAIN |
| SciFact-Open | Secondary | 20 % stratified test split of annotated claims | ~42 claims | SUPPORT / REFUTE |
| CIViC-Fact | Secondary | `partition == "test"`, `flagged != True` | ~2 014 rows | SUPPORT / REFUTE / UNCERTAIN |
