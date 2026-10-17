# experiments/

Entry points, baselines and the config of every reported run. All commands run from the
repository root; CLI flags override config values. Datasets are read from `datasets/`; unpack them first with
`bash scripts/unpack_datasets.sh` (see [../datasets/README.md](../datasets/README.md)).

## ProClaim

SIGNOR uses `run_signor_eval.py`; every other subset uses the generic CSV runner
`run_connectomedb_eval.py`. The direct-mode configs expect a Qwen3.5-9B subagent served by vLLM
at `LLM_BASE_URL`; the outer agent is Claude Sonnet 4.6.

```bash
uv run python experiments/run_signor_eval.py \
    --input-csv datasets/signor.csv --config experiments/configs/signor_direct_config.yaml --reps 3

uv run python experiments/run_connectomedb_eval.py \
    --input-csv datasets/extri.csv --config experiments/configs/extri_direct_config.yaml --reps 1
```

| Setting | Configs |
|---|---|
| Trained sufficiency judge (main results) | `{signor,connectomedb,extri,nubase,nasa_exoplanet}_direct_config.yaml`; `nubase_direct_vertex_config.yaml` and `nasa_exoplanet_direct_vertex_config.yaml` are the same settings through Vertex AI |
| Judge comparison: fixed rounds | `*_direct_fixed_iteration[_vertex]_config.yaml` |
| Judge comparison: always continue | `*_direct_dummy[_vertex]_config.yaml` |
| Judge comparison: random score | `*_direct_random[_vertex]_config.yaml` |
| Judge comparison: Qwen3.5-9B zero-shot | `*_direct_qwen_judge_vertex_config.yaml` |
| Judge comparison: Claude Haiku 4.5 zero-shot | `*_direct_haiku[_vertex]_config.yaml` |
| Judge comparison: Claude Sonnet 4.6 zero-shot | `*_direct_sonnet_vertex_config.yaml` |

The judge comparison is scored on the 40-claim samples (`datasets/<subset>_judge40.csv`). On
ExTRI, NUBASE and NASA every judge, and on all five subsets the Sonnet 4.6 judge, was run on the
sample (`--input-csv datasets/<subset>_judge40.csv`); the other SIGNOR and ConnectomeDB judge runs
covered the full subset and are scored on the sampled claims. No config is included for the
Qwen3.5-9B judge on SIGNOR / ConnectomeDB or for always-continue on ConnectomeDB. The trained judge's weights are in
`results/ablation/models/tau_0.50_seed_42/`.

## Baselines

`run_baselines_datasets.py` runs every baseline except Claude Code and Codex, which have their
own drivers. Biology subsets use PubMed + Semantic Scholar (`pubmed_s2`), NUBASE and NASA use
Semantic Scholar only (`s2`).

```bash
uv run python experiments/run_baselines_datasets.py \
    --config experiments/configs/react_pubmed_s2_max20_config.yaml --datasets extri --seed 100 --repeats 1

uv run python experiments/baselines/claude_code_baseline.py \
    --config experiments/configs/claude_code_pubmed_s2_iclr_config.yaml --dataset extri

uv run python experiments/baselines/codex_baseline.py \
    --config experiments/configs/codex_pubmed_s2_iclr_config.yaml --dataset extri
```

| System | Module | Configs |
|---|---|---|
| Random | `random_baseline.py` | `random_baseline_iclr_config.yaml` (10 draws, seeds 100–109) |
| LLM only (Sonnet 4.6, Gemini 2.5 Flash via `--model`) | `llm_only.py` | `llm_only_config.yaml` |
| Static retrieval, Semantic Scholar top 5 | `retrieval_baseline.py` | `retrieval_config.yaml`; NUBASE: `retrieval_nubase_query_{sonnet,gemini}_vertex_config.yaml` (query = nuclide) |
| ReAct, 10 / 20 / 40 steps | `react_baseline.py` | `react_pubmed_s2_config.yaml`, `react_{pubmed_s2,s2}_max20_config.yaml`, `react_pubmed_s2_max40_config.yaml` |
| ReAct + ProClaim's subclaims, no evidence state | `react_baseline.py` | `react_proclaim_subclaims_{pubmed_s2,s2}_max20_config.yaml` |
| ReAct + subclaims + extracted facts, no evidence state | `react_baseline.py` | `react_proclaim_subclaims_facts_{pubmed_s2,s2}_max20_config.yaml` |
| FIRE | `fire_baseline.py` | `fire_{pubmed_s2,s2}_max20_config.yaml` |
| OpenScholar | `open_scholar_baseline.py` | `open_scholar_config.yaml` |
| Claude Code | `claude_code_baseline.py` | `claude_code_pubmed_s2_iclr_config.yaml`, `claude_code_s2_config.yaml` |
| Codex | `codex_baseline.py` (+ `sonnet_bridge.py`) | `codex_{pubmed_s2,s2}_iclr_config.yaml` |
| One-call evidence representations (abstracts, full text, facts, facts + abstracts) | `oracle_context_baseline.py` | CLI: `--baseline oracle_context --context-mode {papers_only,papers_only_fulltext,facts_only,full} --cached-run-dir <ProClaim run>` |

The subclaim arms read `{dataset}_seed{seed}.json` files written from ProClaim runs by
`scripts/analysis/extract_proclaim_subclaims.py --dataset <subset> --seed <seed> --run-dir <ProClaim run>`
(pass the output folder as `--subclaims-dir`). The fact-extraction arm
also needs the Qwen3.5-9B endpoint at `LLM_BASE_URL`.

FIRE and OpenScholar wrap their upstream code, which is not bundled: clone FIRE into
`experiments/fire` and OpenScholar into `experiments/OpenScholar` (with its own `.venv`).
Claude Code needs `claude-agent-sdk` and Codex needs `openai-codex` (both in `pyproject.toml`).

## Scoring

The paper reports agreement: the fraction of all claims in the subset CSV whose verdict equals
the gold label. `UNCERTAIN`, errored and missing verdicts count as wrong. The accuracy the runners
print excludes errored claims, so it can be higher than agreement on runs with errors.
