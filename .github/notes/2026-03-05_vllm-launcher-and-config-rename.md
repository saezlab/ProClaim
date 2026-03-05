# vLLM Launcher Fixes & Config Field Rename — 2026-03-05

**Branch:** RLM

## Summary

Fixed the HPC vLLM launcher script (`start_vllm_ihpc.sh`) to support configurable GPU types and reliable SSH tunnel reconnection. Renamed the LLM endpoint config fields from `openai_base_url`/`anthropic_base_url` to `subagent_base_url`/`agent_base_url` to make their purpose self-documenting and eliminate confusion about which endpoint serves what role (outer agent vs inner subagent).

Additionally fixed `vllm_node_setup.sh` to auto-select the correct tool-call and reasoning parsers per model family (instead of hardcoding OpenAI parsers), and to launch vLLM inside the Singularity container (the bare `vllm` binary doesn't exist on compute nodes). Also made the SLURM allocation polling loop tolerant of transient SSH failures, fixed `EvidenceState.save()` to accept optional path arguments, and added auto-save guidance to the SYSTEM_PROMPT.

## Modified Files

| File | Changes |
|------|---------|
| `start_vllm_ihpc.sh` (outside repo) | Added `--gpu-type` flag (default `a100`), interpolated into SLURM `--gres`; fixed `--reconnect` and `--logs` to use direct SSH instead of multiplexed connections that caused premature tunnel death; made SLURM polling loop tolerate up to 3 consecutive SSH failures before giving up |
| `vllm_node_setup.sh` (outside repo) | Added `case` statement on model name to auto-select `--tool-call-parser` and `--reasoning-parser` per model family; wrapped `vllm serve` in `singularity exec --nv` (bare `vllm` binary doesn't exist on compute nodes) |
| `src/pkevolve/verification/evidence_state.py` | Made `save(path)` argument optional — defaults to `_workspace/evidence_state.json`; accepts directories (auto-appends `evidence_state.json`) |
| `src/pkevolve/verification/config.py` | Renamed `anthropic_base_url` → `agent_base_url`, `openai_base_url` → `subagent_base_url`; changed `alias` to `validation_alias` + `populate_by_name=True` to fix pydantic-settings priority bug where YAML values were silently ignored; updated properties, CLI arg parser, and override merging |
| `src/pkevolve/verification/repl_orchestrator.py` | Updated `cfg.openai_base_url` → `cfg.subagent_base_url` |
| `scripts/verification/demo_evidence_programming.py` | Updated system prompt template to use `cfg.subagent_base_url`; added auto-save guidance to SYSTEM_PROMPT (tells agent not to call `state.save()` manually) |
| `experiments/example_config.yaml` | Renamed YAML keys to `subagent_base_url` / `agent_base_url` |
| `doc/sdk_vs_repl_modes.md` | Updated config example to use new field names |

## Key Design Decisions

- **`validation_alias` instead of `alias`**: pydantic-settings `alias` replaces the field name entirely for construction, which means YAML keys using the field name (`subagent_base_url`) are silently ignored. `validation_alias` keeps the field name usable for both Python attribute access and constructor kwargs, while still allowing env vars (`LLM_BASE_URL`, `ANTHROPIC_BASE_URL`) as fallback inputs.

- **`populate_by_name=True`**: Required alongside `validation_alias` so that pydantic accepts both the field name (from YAML/code) and the validation alias (from env vars) during construction. Without it, only the alias name works.

- **Direct SSH for `--reconnect`/`--logs`**: The SSH multiplexing (`ControlMaster=auto`, `ControlPersist=600`) caused the `ssh -N` tunnel to return immediately because the ControlMaster handled the connection and the `-N` client had nothing to wait for. Switched to plain direct SSH with `ServerAliveInterval=30` keepalives.

- **No auto-redirect for localhost URLs**: Considered auto-detecting the vLLM server hostname from `.vllm_server_info`, but decided users should set explicit endpoints in config. The field rename makes the purpose clear enough to avoid misconfiguration.

- **Model-aware parser selection in `vllm_node_setup.sh`**: Instead of hardcoding `--tool-call-parser openai --reasoning-parser openai_gptoss` (which causes 500 errors with non-OpenAI models), added a `case` on `$MODEL_LOWER` that maps model families to their correct parsers: Qwen3/3.5/QwQ → `hermes` + `qwen3`, Qwen3-Coder → `qwen3_xml` + `qwen3`, Qwen2 → `hermes`, GLM-4.5/4.6 → `glm45`, GLM-4.7 → `glm47`, DeepSeek R1 → `deepseek_v3` + `deepseek_r1`, Llama 3 → `llama3_json`, Llama 4 → `llama4_pythonic`, Mistral → `mistral`. Unknown models get a warning and serve without parser flags.

- **Singularity exec wrapping**: The compute nodes don't have `vllm` installed — it only exists inside the Singularity image. The `vllm serve` command must be wrapped with `singularity exec --nv $IMAGE`.

- **SSH failure tolerance in polling loop**: During long SLURM queue waits, the SSH multiplexed connection can drop due to network hiccups (unrelated to the compute node starting). The polling loop now tolerates up to 3 consecutive SSH check failures before declaring the job dead, and does a final info-file check before exiting.

## Bug Fixes

- **SSH tunnel dying on `--reconnect`**: The SSH multiplexing options caused the tunnel client to exit immediately after the ControlMaster took over. Fixed by using direct SSH connections for the reconnect and logs actions.

- **YAML config values silently ignored**: When `LLMSettings` used `alias="LLM_BASE_URL"`, pydantic-settings only accepted `LLM_BASE_URL` as a constructor kwarg, not `subagent_base_url`. This meant YAML values for the endpoint field were dropped and the default (`localhost:8000`) was always used — the root cause of the LLM connection failures in generated notebooks. Fixed with `validation_alias` + `populate_by_name=True`.

- **Hardcoded GPU type**: The SLURM `--gres` flag was hardcoded to `gpu:a100:N`. Added `--gpu-type` CLI flag so other GPU models (e.g., H100) can be requested.

- **Hardcoded OpenAI parsers caused 500 errors**: `vllm_node_setup.sh` hardcoded `--tool-call-parser openai --reasoning-parser openai_gptoss`, which crashes vLLM when serving Qwen models (the OpenAI gpt-oss reasoning parser can't handle `<think>` blocks). Fixed with model-family-aware parser selection via `case` statement.

- **Missing Singularity wrapping**: `vllm serve` was called directly but the binary only exists inside the Singularity image (`vllm-openai_latest.sif`). Wrapped with `singularity exec --nv`.

- **Transient SSH failures killed the launcher**: A single SSH check failure during SLURM allocation wait would cause the launcher to exit with "SLURM job failed", even though the job was still pending. Added tolerance for up to 3 consecutive failures.

- **`EvidenceState.save()` TypeError**: The SDK agent sometimes called `state.save()` without arguments, or `state.save(workspace)` where workspace is a directory. Made `path` optional (defaults to `_workspace/evidence_state.json`) and added directory detection (auto-appends `evidence_state.json`).

- **Agent unnecessarily calling `save()`**: The SYSTEM_PROMPT now explicitly tells the agent that persistence is handled automatically and manual `save()` calls are unnecessary.
