# Upstream Baselines And NeurIPS Analysis — 2026-05-02

**Branch:** `exp`

## Summary

Recent work updated both the baseline evaluation stack and the sandbox-vs-wild analysis tooling. On the baseline side, FIRE and SAFE were switched from local reimplementations to thin wrappers around the upstream repositories, OpenScholar was pointed at its in-repo checkout, deprecated FIRE+S2 paths were marked as historical-only, and the SLURM runner was updated to reflect the supported baseline set and shared datasets location. On the analysis side, the transition plotting script was expanded to merge SIGNOR-Fact and ConnectomeDB-Fact by default, preserve the legacy top-level output path for aggregate runs, switch the third setting from the older `s2_plus_ref` baseline to ProClaim direct-eval outputs, and generate a simpler three-panel NeurIPS-oriented consensus figure in addition to the existing alluvial and confusion-matrix outputs.

## New Files

| File | Purpose |
|------|---------|
| `experiments/baselines/shared/upstream_adapters.py` | Provides the adapter layer for running upstream FIRE and SAFE modules with the repo's shared `LLMBackend` and shared web-search helper. |

## Modified Files

| File | Changes |
|------|---------|
| `experiments/baselines/fire_baseline.py` | Replaced the local FIRE loop with an upstream FIRE wrapper, restricted the integration to web search, and routed prompt/search logging through the adapter-backed execution path. |
| `experiments/baselines/safe_baseline.py` | Replaced the local SAFE loop with an upstream SAFE wrapper, restricted the integration to web search, and recorded prompt/search usage through the adapter-backed execution path. |
| `experiments/baselines/open_scholar_baseline.py` | Corrected the default OpenScholar repository root to `experiments/OpenScholar`. |
| `experiments/configs/fire_s2_config.yaml` | Marked the FIRE+S2 preset as deprecated historical configuration that now fails fast if used unchanged. |
| `experiments/README.md` | Updated baseline documentation to reflect the upstream FIRE/SAFE integrations, the web-only constraint for those baselines, and the deprecated FIRE+S2 variant. |
| `scripts/run_all_baselines_slurm.sh` | Removed `fire_s2` from the submitted baseline set, added the shared `--datasets-dir` path, and updated the help text and numbering accordingly. |
| `scripts/analysis/plot_sandbox_vs_wild_transitions.py` | Added multi-dataset loading for SIGNOR and ConnectomeDB, restored top-level aggregate outputs, switched Setting 3 defaults from `s2_plus_ref` JSONL outputs to ProClaim direct-eval JSONLs, renamed Setting 3 captions accordingly, retained the triple confusion-matrix figure, and added a separate three-panel NeurIPS consensus summary figure with setting-specific captions and tighter paper-scale typography. |
| `scripts/analysis/convert_direct_eval_csv_to_jsonl.py` | Added a converter from SIGNOR direct-eval CSV outputs to baseline-style `BaselineResult` JSONL using `SIGNOR_ID` / `SIGNOR_ID_flip` claim IDs so ProClaim outputs can be compared against the existing `single_paper` and retrieval JSONL files. |

## Architecture

```text
Baseline evaluation path

scripts/run_all_baselines_slurm.sh
  -> experiments/run_baselines_datasets.py
     -> FIREBaseline / SAFEBaseline / OpenScholarBaseline
        -> experiments/baselines/shared/upstream_adapters.py
           -> upstream FIRE / SAFE modules
           -> shared do_search web helper

Analysis path

results/baselines/single_paper/.../{signor,connectomedb}_seed100.jsonl
   -> Setting 1 (single-passage context)

results/baselines/retrieval/s2/.../top5/{signor,connectomedb}_seed100.jsonl
   -> Setting 2 (naive retrieval)

results/baselines/{signor_direct_eval_20260427_221617,connectomedb_eval_20260424_171117}/*.jsonl
   -> Setting 3 (ProClaim)

  -> scripts/analysis/plot_sandbox_vs_wild_transitions.py
     -> load_and_merge(signor, connectomedb)
     -> compute transitions / accuracies / confusion matrices
     -> save alluvial, confusion, and NeurIPS summary figures to results/analysis
```

## Key Design Decisions

- FIRE and SAFE now follow the upstream repositories' control flow and prompts instead of maintaining local approximations, reducing drift from the reference implementations.
- The upstream integrations are web-only because the imported FIRE and SAFE paths do not expose a Semantic Scholar mode compatible with the earlier local wrappers.
- The patched upstream search hook calls the repo's shared `do_search` helper so FIRE and SAFE reuse the same web retrieval path as the rest of the baseline framework.
- The aggregate transition analysis keeps writing to `results/analysis` for the default combined run so reruns overwrite the legacy top-level artifacts rather than leaving stale SIGNOR-only figures in place.
- Setting 3 in the NeurIPS summary figure now points to ProClaim direct-eval outputs because the paper-facing comparison is intended to show the full agentic system rather than the earlier reference-plus-retrieval baseline.
- The NeurIPS summary figure uses one panel per setting and normalized within-row prediction shares to make the message readable in under a minute without requiring the reader to decode a dense confusion matrix.
- The second and third panels hide repeated left-side verdict labels to reduce visual redundancy, and the figure-level title remains optional so the panel headers and legend have enough vertical room at paper scale.

## Figure Takeaway

The combined SIGNOR-Fact + ConnectomeDB-Fact figure is meant to support a three-step motivation argument.

1. Document-guided verification is not enough to approximate consensus. With only the cited reference abstract, prediction-label agreement reaches 58.5%, so a single source document is too narrow to stand in for the broader literature.
2. Naive retrieval is necessary but still insufficient. Replacing the reference abstract with five retrieved abstracts drops agreement to 50.8% and causes substantial regression on gold-SUPPORT claims. The dominant failure mode is over-prediction of `REFUTE`.
3. ProClaim improves this failure mode by performing adaptive retrieval, evidence organization, and reconciliation before deciding, raising prediction-label agreement to 73.7%.

The key interpretation is that the Settings 1 and 2 `REFUTE` skew is not a plotting artifact and not simply a consequence of class balance. The underlying baselines use the same one-shot verdict prompt and differ only in the evidence source. Under the current verdict definitions, `REFUTE` covers both direct contradiction and the case where a thorough search fails to find substantiating evidence. With only one abstract or a noisy top-5 retrieved set, many true `SUPPORT` claims never receive an explicit positive statement of the target interaction, so the system collapses absence of explicit support into `REFUTE` rather than `UNCERTAIN`. This is the main reason both non-agentic settings overproduce `REFUTE` predictions.

## Paper Narrative

Suggested NeurIPS paragraph:

> Using the SIGNOR-Fact dataset, Figure~\ref{fig:sandbox_vs_wild} makes the case for in-the-wild verification. **(i) Document-guided verification is insufficient for consensus:** when the model sees only the reference abstract, prediction-label agreement is just 58.5\%, showing that a single cited document is not a reliable proxy for the broader scientific consensus. **(ii) Naive retrieval is necessary, but still insufficient:** replacing the reference abstract with five retrieved abstracts reduces agreement to 50.8\%, and 22\% of \textsc{support} claims regress relative to the document-guided setting. The dominant failure mode in both settings is over-prediction of \textsc{refute}. This is not because the benchmark is refute-heavy, but because limited or noisy evidence often fails to contain an explicit positive statement of the claim, and our verdict definitions map this absence of substantiating evidence to \textsc{refute} rather than \textsc{uncertain} (Appendix~\ref{app:verdict_definitions}). **(iii) ProClaim addresses this failure mode:** with adaptive retrieval, evidence organisation, and reconciliation before rendering a final verdict, ProClaim raises prediction-label agreement to 73.7\%. Together, these results show that in-the-wild verification requires knowledge construction over retrieved evidence, not one-shot decisions from either a single document or a naively pooled set of abstracts.

## Bug Fixes

- Fixed the aggregate sandbox-vs-wild plotting workflow so the default combined SIGNOR+ConnectomeDB run updates the expected top-level analysis outputs instead of leaving stale single-dataset figures visible there.
- Corrected the default OpenScholar checkout path from a workspace-level location to the in-repo `experiments/OpenScholar` directory.
- Removed `fire_s2` from the active SLURM baseline submission path while keeping the config as a documented historical artifact, preventing silent submission of an unsupported variant.
- Removed stale Setting 3 wording that still referred to retrieved-plus-reference evidence after the figure switched to ProClaim inputs.
- Fixed repeated y-axis category labels on the second and third panels of the NeurIPS summary figure.
- Simplified the NeurIPS panel captions to one-line setting descriptions so the explanatory text remains readable at presentation size.
- Resolved multiple layout collisions in the summary figure header by separating the panel title, short description, and prediction-label agreement text and then rebalancing panel width and inter-panel spacing.