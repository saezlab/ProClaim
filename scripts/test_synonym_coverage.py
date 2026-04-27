"""
Test: does expanding state.claim with entity synonyms improve entity coverage and NLI scores?

Uses the same populate_paper_features + MLP pipeline as the real run.
- NLP features recomputed for 5 representative papers (the ones with facts).
- Remaining papers keep their existing features from the saved state.
- MLP sufficiency score is then computed on the full 63-paper set.
"""

import sys, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pkevolve.verification.evidence_api import setup_workspace, populate_paper_features

WORKSPACE = str(ROOT / "results/test_direct_1/workspace")

# Papers with actual facts / relevant text — recompute features for these
TEST_PMIDS = ["9153252", "15169891", "18353971", "12060669", "35562995"]

# Which test papers were present in each search round (from execution_log.py)
ROUND_PMIDS = {
    "R1 (36 papers)": ["12060669", "35562995"],
    "R2 (60 papers)": ["12060669", "35562995", "15169891", "18353971"],
    "R3 (63 papers)": ["12060669", "35562995", "15169891", "18353971", "9153252"],
}

CLAIMS = {
    "A_original":  "SRC directly inhibits CTTN.",
    "B_paren_syn": "SRC (c-Src, pp60-Src) directly inhibits CTTN (cortactin, EMS1).",
    "C_slash_syn": "SRC/c-Src directly inhibits CTTN/cortactin.",
    "D_full_syn":  (
        "SRC (c-Src, pp60-Src, proto-oncogene tyrosine-protein kinase Src, pp60c-src, v-src) "
        "directly inhibits CTTN (cortactin, EMS1, HS1, p80 cortactin)."
    ),
}


def compute_mlp_score(state, pmid_subset: list[str] | None = None) -> tuple[float, dict]:
    """Run MLP on papers in state (optionally filtered to pmid_subset)."""
    from pkevolve.verification.model_registry import get_mlp_classifier, get_feature_aggregator
    from sufficiency_classifier.test_mlp_classifier import flatten_features, mlp_predict

    mlp = get_mlp_classifier()
    aggregator = get_feature_aggregator()

    papers = state.papers
    if pmid_subset is not None:
        papers = {k: v for k, v in papers.items() if k in pmid_subset}

    papers_dicts = []
    num_full_text = 0
    for paper in papers.values():
        d: dict = {}
        if paper.metadata:
            d["metadata_features"] = paper.metadata.model_dump()
        if paper.nlp:
            d["nlp_features"] = paper.nlp.model_dump()
        papers_dicts.append(d)
        if paper.full_text:
            num_full_text += 1

    nested = aggregator.aggregate_all(papers_dicts)
    flat = flatten_features(nested)
    flat["num_full_text"] = float(num_full_text)

    prob, _ = mlp_predict(mlp["model"], flat, mlp["expected_features"], mlp["mean"], mlp["scale"])
    return prob, flat


def run_for_claim(base_state, label: str, claim: str):
    """Returns (state_with_recomputed_features, per_paper_nlp_dict).
    The returned state can be re-sliced by round for MLP scoring.
    """
    state = copy.deepcopy(base_state)
    state.claim = claim

    for pmid in TEST_PMIDS:
        if pmid in state.papers:
            state.papers[pmid].nlp = None
            state.papers[pmid].metadata = None

    populate_paper_features(state, compute_nli=True, max_text_length=5000, force_recompute=False)

    per_paper = {
        pmid: {
            "entity_coverage": state.papers[pmid].nlp.claim_entity_coverage if state.papers[pmid].nlp else None,
            "semantic_sim":    state.papers[pmid].nlp.semantic_similarity    if state.papers[pmid].nlp else None,
            "nli_entail":      state.papers[pmid].nlp.nli_entailment         if state.papers[pmid].nlp else None,
            "nli_contra":      state.papers[pmid].nlp.nli_contradiction      if state.papers[pmid].nlp else None,
            "claim_ents":      state.papers[pmid].nlp.claim_entities         if state.papers[pmid].nlp else [],
            "evid_ents":       state.papers[pmid].nlp.evidence_entities      if state.papers[pmid].nlp else [],
        }
        for pmid in TEST_PMIDS
        if pmid in state.papers
    }

    return state, per_paper


def fmt(v):
    return f"{v:.3f}" if v is not None else "  N/A"


def main():
    print("Loading base state …")
    base_state, _llm, _ws = setup_workspace(
        claim=CLAIMS["A_original"], workspace_path=WORKSPACE
    )

    # -------------------------------------------------------------------------
    # Compute NLP features per claim variant
    # -------------------------------------------------------------------------
    states:     dict[str, object] = {}
    all_per_paper: dict[str, dict] = {}

    for label, claim in CLAIMS.items():
        print(f"\n>>> Claim [{label}]: {claim}")
        st, per_paper = run_for_claim(base_state, label, claim)
        states[label]       = st
        all_per_paper[label] = per_paper

    # -------------------------------------------------------------------------
    # Claim entity sets
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CLAIM ENTITY SETS (scispaCy)")
    print("=" * 80)
    for lbl in CLAIMS:
        ents = all_per_paper[lbl].get(TEST_PMIDS[0], {}).get("claim_ents", [])
        print(f"  [{lbl}]  {ents}")

    # -------------------------------------------------------------------------
    # Per-paper NLP table
    # -------------------------------------------------------------------------
    METRICS = ["entity_coverage", "nli_entail", "nli_contra", "semantic_sim"]
    labels  = list(CLAIMS.keys())
    col_w   = 13

    print("\n" + "=" * 80)
    print("PER-PAPER NLP FEATURES (5 test papers, recomputed per variant)")
    print("=" * 80)

    for pmid in TEST_PMIDS:
        p = base_state.papers.get(pmid)
        title = (p.title or pmid) if p else pmid
        print(f"\nPMID {pmid}: {title[:72]}")
        print(f"  {'metric':<18}", end="")
        for lbl in labels:
            print(f"  {lbl:>{col_w}}", end="")
        print()
        print(f"  {'':-<18}", end="")
        for _ in labels:
            print(f"  {'':->{col_w}}", end="")
        print()
        for m in METRICS:
            print(f"  {m:<18}", end="")
            for lbl in labels:
                v = all_per_paper[lbl].get(pmid, {}).get(m)
                print(f"  {fmt(v):>{col_w}}", end="")
            print()

    # -------------------------------------------------------------------------
    # MLP score matrix: 3 rounds × 3 claim variants
    # Note: only test papers are in the current state; other papers in the
    # original run are no longer available. Scores reflect the test-paper
    # subset only (the papers that actually drove NLI/coverage signals).
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MLP SUFFICIENCY SCORE — 3 rounds × 3 claim variants")
    print("(using test-paper subsets that were present in each round)")
    print("=" * 80)

    round_labels = list(ROUND_PMIDS.keys())
    claim_labels = list(CLAIMS.keys())

    # Header
    print(f"\n  {'round':<22}", end="")
    for cl in claim_labels:
        print(f"  {cl:>14}", end="")
    print()
    print(f"  {'':-<22}", end="")
    for _ in claim_labels:
        print(f"  {'':-<14}", end="")
    print()

    # Cells
    all_scores: dict[str, dict[str, float]] = {}
    for rl in round_labels:
        subset = ROUND_PMIDS[rl]
        print(f"  {rl:<22}", end="")
        all_scores[rl] = {}
        for cl in claim_labels:
            prob, _ = compute_mlp_score(states[cl], pmid_subset=subset)
            all_scores[rl][cl] = prob
            label_str = "suf" if prob >= 0.5 else "insuf"
            print(f"  {prob:.4f} ({label_str})", end="")
        print()

    # -------------------------------------------------------------------------
    # Key aggregated features for R3 (most complete) per claim variant
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("KEY MLP INPUT FEATURES — R3 (all 5 papers) per claim variant")
    print("=" * 80)
    key_feats = [
        "max_entity_coverage", "mean_entity_coverage",
        "max_similarity",      "mean_similarity",
        "entailment_ratio",    "contradiction_ratio",
        "controversy_index",   "num_papers",
    ]
    print(f"\n  {'feature':<28}", end="")
    for cl in claim_labels:
        print(f"  {cl:>14}", end="")
    print()
    print(f"  {'':-<28}", end="")
    for _ in claim_labels:
        print(f"  {'':-<14}", end="")
    print()
    for feat in key_feats:
        print(f"  {feat:<28}", end="")
        for cl in claim_labels:
            _, flat = compute_mlp_score(states[cl], pmid_subset=ROUND_PMIDS["R3 (63 papers)"])
            v = flat.get(feat)
            print(f"  {fmt(v):>14}", end="")
        print()

    # -------------------------------------------------------------------------
    # Gain summary vs A_original
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MLP PROB GAIN vs A_original")
    print("=" * 80)
    for cl in claim_labels[1:]:
        print(f"\n  [{cl}]")
        for rl in round_labels:
            diff = all_scores[rl][cl] - all_scores[rl]["A_original"]
            sign = "+" if diff >= 0 else ""
            print(f"    {rl}: {sign}{diff:.4f}")


if __name__ == "__main__":
    main()
