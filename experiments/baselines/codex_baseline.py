"""Codex CLI as a black-box agentic baseline, via the ``openai_codex`` SDK.

Mirrors ``claude_code_baseline.py`` (same prompt, labels and result schema) so
the two harnesses are directly comparable. Runs on Claude Sonnet-4-6, not an
OpenAI model: Codex speaks only OpenAI wire formats, so traffic is routed
through ``sonnet_bridge.py``, launched and torn down here as a subprocess.
``CODEX_HOME`` is redirected to a scratch dir, which is git-initialized at
startup (Codex requires a git repo).

Usage::

    uv run python experiments/baselines/codex_baseline.py --dataset signor --limit 5
    uv run python experiments/baselines/codex_baseline.py --dataset signor \\
        --config experiments/configs/codex_pubmed_s2_iclr_config.yaml
"""

from __future__ import annotations

import argparse
import json
import asyncio
import atexit
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from proclaim.verification.prompts import (
    CLAUDE_CODE_BASELINE_SYSTEM_PROMPT,
    CLAUDE_CODE_BASELINE_USER_PROMPT,
    PUBMED_S2_TOOL_HINT_SHELL,
    S2_TOOL_HINT_SHELL,
)

from baselines.shared.claim_loader import load_claim_csv
from baselines.shared.cost_tracker import CostTracker
from baselines.shared.evaluate import EvaluationHarness
from baselines.shared.label_utils import get_label_config, normalize_label
from baselines.shared.logging_utils import write_claim_log
from baselines.shared.verdict import BaselineResult, extract_json

from openai_codex import AsyncCodex, CodexConfig, Sandbox

logger = logging.getLogger(__name__)

DEFAULT_DATASETS_DIR = Path("datasets")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "baselines" / "codex"
BRIDGE_SCRIPT = Path(__file__).parent / "sonnet_bridge.py"
BRIDGE_MODEL_PROVIDER = "sonnet_bridge"
BRIDGE_MODEL = "claude-sonnet-4-6"
EXPERIMENTS_DIR = str(PROJECT_ROOT / "experiments")

LC = get_label_config()


# Evidence sources served by ProClaim's own search backend through shell calls.
SHELL_TOOL_HINTS = {"pubmed_s2": PUBMED_S2_TOOL_HINT_SHELL, "s2": S2_TOOL_HINT_SHELL}


def build_system_prompt(evidence_source: str) -> str:
    if evidence_source in SHELL_TOOL_HINTS:
        tool_hint = SHELL_TOOL_HINTS[evidence_source].format(
            python_exe=sys.executable, experiments_dir=EXPERIMENTS_DIR,
        )
    else:
        tool_hint = ""
    return CLAUDE_CODE_BASELINE_SYSTEM_PROMPT.format(
        verdict_names=", ".join(LC.verdict_names()),
        verdict_definitions=LC.verdict_prompt_block(),
        label_options=" | ".join(f'"{n}"' for n in LC.verdict_names()),
        tool_hint=tool_hint,
    )


class CodexBaseline:
    """Verify claims with the real Codex CLI, driven as a black box."""

    name = "codex"

    def __init__(
        self,
        *,
        dataset: str = "",
        evidence_source: str = "web",
        timeout_seconds: float = 900,
        bridge_port: int = 4180,
        scratch_dir: Path | str | None = None,
        bridge_model: str | None = None,
        bridge_vertex_location: str | None = None,
        bridge_extra_body: dict | None = None,
    ) -> None:
        self.dataset = dataset
        self.evidence_source = evidence_source
        self.timeout_seconds = timeout_seconds
        self.bridge_port = bridge_port
        # bridge_model: LiteLLM target served by sonnet_bridge.py (unset = its
        # default, Sonnet 4.6 on Vertex). Codex sees the name without the provider.
        self.bridge_model = bridge_model
        self.bridge_vertex_location = bridge_vertex_location
        self.bridge_extra_body = bridge_extra_body
        self.model = bridge_model.split("/", 1)[-1] if bridge_model else BRIDGE_MODEL
        self.log_dir: Path | None = None

        scratch = Path(scratch_dir) if scratch_dir else Path(tempfile.mkdtemp(prefix=f"{self.name}_"))
        self.codex_home = scratch / "codex_home"
        self.workdir = scratch / "workdir"
        self.bridge_log = scratch / "sonnet_bridge.log"
        self.codex_home.mkdir(parents=True, exist_ok=True)
        self.workdir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Isolated Codex session dir (never touches the real ~/.codex): %s",
            self.codex_home,
        )
        self._init_git_workdir()

        # Codex reports no dollar cost, so it's computed from the shared table.
        self._in_price, self._out_price = CostTracker.pricing_for(bridge_model or BRIDGE_MODEL)
        self._bridge: subprocess.Popen | None = None
        self._codex: AsyncCodex | None = None

    def _init_git_workdir(self) -> None:
        """Codex requires a git repo by default."""
        if (self.workdir / ".git").exists():
            return
        for cmd in (
            ["git", "init", "-q"],
            ["git", "config", "user.email", "codex-baseline@example.com"],
            ["git", "config", "user.name", "codex-baseline"],
            ["git", "commit", "--allow-empty", "-q", "-m", "init"],
        ):
            subprocess.run(cmd, cwd=self.workdir, check=True)


    def start_bridge(self) -> None:
        """Launch sonnet_bridge.py and wait for it to become healthy."""
        if self._bridge is not None:
            return
        logger.info("Starting sonnet_bridge on port %d (log: %s)", self.bridge_port, self.bridge_log)
        log_fh = open(self.bridge_log, "w")
        env = dict(os.environ)
        if self.bridge_model:
            env["SONNET_BRIDGE_MODEL"] = self.bridge_model
        if self.bridge_vertex_location:
            env["SONNET_BRIDGE_VERTEX_LOCATION"] = self.bridge_vertex_location
        if self.bridge_extra_body:
            env["SONNET_BRIDGE_EXTRA_BODY"] = json.dumps(self.bridge_extra_body)
        proc = subprocess.Popen(
            [sys.executable, str(BRIDGE_SCRIPT), "--port", str(self.bridge_port)],
            stdout=log_fh, stderr=subprocess.STDOUT, env=env,
        )
        url = f"http://127.0.0.1:{self.bridge_port}/v1/models"
        for _ in range(60):
            if proc.poll() is not None:
                raise RuntimeError(f"sonnet_bridge died before starting; see {self.bridge_log}")
            try:
                urllib.request.urlopen(url, timeout=2)
                self._bridge = proc
                atexit.register(self.stop_bridge)
                logger.info("sonnet_bridge is healthy.")
                return
            except (urllib.error.URLError, ConnectionError):
                time.sleep(1)
        proc.kill()
        raise RuntimeError(f"sonnet_bridge did not become healthy in time; see {self.bridge_log}")

    def stop_bridge(self) -> None:
        if self._bridge is None:
            return
        self._bridge.terminate()
        try:
            self._bridge.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._bridge.kill()
        self._bridge = None

    def _config(self) -> CodexConfig:
        overrides = [
            f"model_providers.{BRIDGE_MODEL_PROVIDER}.name=Sonnet Bridge",
            f"model_providers.{BRIDGE_MODEL_PROVIDER}.base_url=http://127.0.0.1:{self.bridge_port}/v1",
            f"model_providers.{BRIDGE_MODEL_PROVIDER}.env_key=CODEX_BRIDGE_KEY",
            f"model_providers.{BRIDGE_MODEL_PROVIDER}.wire_api=responses",
        ]
        if self.evidence_source in SHELL_TOOL_HINTS:
            # System prompt forces PubMed/S2 shell calls instead.
            overrides.append("web_search=disabled")
        else:
            overrides.append("web_search=live")
        return CodexConfig(
            env={
                **os.environ,
                "CODEX_HOME": str(self.codex_home),
                "CODEX_BRIDGE_KEY": "dummy-key-not-checked",
            },
            config_overrides=tuple(overrides),
        )

    async def __aenter__(self) -> CodexBaseline:
        """Share one bridge + Codex client across a whole run."""
        self.start_bridge()
        self._codex = await AsyncCodex(self._config()).__aenter__()
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._codex is not None:
            await self._codex.__aexit__(*exc_info)
            self._codex = None
        self.stop_bridge()

    def _sandbox(self) -> Sandbox:
        # pubmed_s2 / s2 need real shell exec; workspace_write's bwrap sandbox
        # crashes here with `pivot_root: Invalid argument`, so bypass it.
        return Sandbox.full_access if self.evidence_source in SHELL_TOOL_HINTS else Sandbox.workspace_write


    async def averify(
        self,
        claim_id: str,
        claim: str,
        gold_label: str,
        *,
        context: dict | None = None,
    ) -> BaselineResult:
        del context
        if self._codex is not None:
            return await self._run_one(self._codex, claim_id, claim, gold_label)
        # Standalone call (sequential runner): own client for this claim.
        self.start_bridge()
        async with AsyncCodex(self._config()) as codex:
            return await self._run_one(codex, claim_id, claim, gold_label)

    async def _run_one(
        self,
        codex: AsyncCodex,
        claim_id: str,
        claim: str,
        gold_label: str,
    ) -> BaselineResult:
        user_prompt = CLAUDE_CODE_BASELINE_USER_PROMPT.format(claim=claim)

        t0 = time.monotonic()
        final_text = ""
        tool_calls: list[str] = []
        input_tokens = output_tokens = cached_tokens = 0
        error_msg = ""

        try:
            thread = await codex.thread_start(
                developer_instructions=build_system_prompt(self.evidence_source),
                cwd=str(self.workdir),
                model=self.model,
                model_provider=BRIDGE_MODEL_PROVIDER,
                sandbox=self._sandbox(),
            )
            result = await asyncio.wait_for(thread.run(user_prompt), timeout=self.timeout_seconds)

            final_text = result.final_response or ""
            for item in result.items:
                node = item.root if hasattr(item, "root") else item
                item_type = getattr(node, "type", None)
                if item_type in ("agentMessage", "userMessage", "reasoning"):
                    continue
                # Server-executed web_search is stripped by sonnet_bridge.py,
                # so real web searches never appear in `evidence`.
                if item_type == "webSearch":
                    tool_calls.append(f"webSearch:{getattr(node, 'query', '')[:60]}")
                elif item_type == "commandExecution":
                    tool_calls.append(f"shell:{getattr(node, 'command', '')[:60]}")
                elif item_type:
                    tool_calls.append(item_type)

            if result.usage is not None:
                input_tokens = result.usage.total.input_tokens
                output_tokens = result.usage.total.output_tokens
                cached_tokens = result.usage.total.cached_input_tokens
        except asyncio.TimeoutError:
            # str(TimeoutError()) is "" — spell it out so the row isn't
            # mistaken for a real verdict.
            error_msg = f"TIMEOUT: no response after {self.timeout_seconds:.0f}s"
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc) or f"{type(exc).__name__} (no message)"

        elapsed = time.monotonic() - t0
        cost = (
            (input_tokens / 1e6) * self._in_price
            + (output_tokens / 1e6) * self._out_price
        )
        parsed = extract_json(final_text)
        # Nothing parseable after a failure is an ERROR, not UNCERTAIN. A label
        # surviving alongside a non-fatal error still counts as a verdict.
        failed = bool(error_msg) and not parsed.get("label")
        predicted = normalize_label(parsed.get("label")) if parsed.get("label") else "UNCERTAIN"
        reasoning = parsed.get("reasoning", "") or error_msg
        if failed:
            predicted = "ERROR"
            logger.error("%s produced no verdict: %s", claim_id, error_msg)

        write_claim_log(
            self.log_dir,
            claim_id,
            [
                (f"claim_id: {claim_id}", ""),
                ("claim", claim),
                (f"evidence_source: {self.evidence_source}  sandbox: {self._sandbox()}", ""),
                (f"tokens in/out/cached: {input_tokens}/{output_tokens}/{cached_tokens}", ""),
                (f"tool calls ({len(tool_calls)})", "\n".join(tool_calls)),
                ("final response", final_text),
                ("error", error_msg),
            ],
        )

        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=predicted,
            reasoning=reasoning[:1000],
            evidence=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_seconds=elapsed,
            baseline_name=self.name,
            model=self.model if not error_msg else "",
            dataset=self.dataset,
            error=error_msg if failed else "",
        )

    def verify(
        self,
        claim_id: str,
        claim: str,
        gold_label: str,
        *,
        context: dict | None = None,
    ) -> BaselineResult:
        """Blocking entry point, for the shared sequential harness/runner."""
        return asyncio.run(self.averify(claim_id, claim, gold_label, context=context))


async def main_async(args: argparse.Namespace) -> None:
    claims = load_claim_csv(args.dataset, args.datasets_dir, args.limit)
    logger.info("Loaded %d claims for dataset=%s", len(claims), args.dataset)

    baseline = CodexBaseline(
        dataset=args.dataset,
        evidence_source=args.evidence_source,
        timeout_seconds=args.timeout_seconds,
        bridge_port=args.bridge_port,
        scratch_dir=args.scratch_dir,
        bridge_model=args.bridge_model,
        bridge_vertex_location=args.bridge_vertex_location,
        bridge_extra_body=args.bridge_extra_body,
    )
    harness = EvaluationHarness(baseline, dataset_name=args.dataset)

    output_dir = Path(args.output_dir)
    stem = f"{args.dataset}_{args.evidence_source}"
    output_path = output_dir / f"{stem}.jsonl"

    t_start = time.monotonic()
    async with baseline:
        await harness.arun(claims, resume_path=output_path, concurrency=args.concurrency)
    logger.info("Done. Total wall time: %.1f min", (time.monotonic() - t_start) / 60)

    metrics = EvaluationHarness.metrics(EvaluationHarness.load(output_path))
    EvaluationHarness.save_metrics(metrics, output_dir / f"{stem}_metrics.json")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=None,
        help="YAML config supplying defaults for any flag below (see "
             "experiments/configs/codex_*_config.yaml). Keys are the flags' "
             "long names with dashes as underscores, e.g. `timeout_seconds`. "
             "Any flag passed on the command line overrides the config.",
    )
    # Validated after parsing, not via required=True, so --config can supply it.
    # No `choices=`: the loader reads {dataset}.csv from --datasets-dir, and the
    # prompt depends only on --evidence-source (as in claude_code_baseline.py).
    parser.add_argument("--dataset", default=None,
                        help="Dataset stem; {dataset}.csv must exist in --datasets-dir.")
    parser.add_argument("--datasets-dir", default=str(DEFAULT_DATASETS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scratch-dir", default=None,
                         help="Scratch dir for isolated Codex session state + git workdir. "
                              "Defaults to a fresh temp dir (auto-cleaned by the OS).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=900,
                         help="Per-claim wall-clock timeout (Codex has no native max-turns/"
                              "budget cap like Claude Code, so this is the safety net).")
    parser.add_argument("--bridge-port", type=int, default=4180)
    parser.add_argument("--bridge-model", default=None,
                        help="LiteLLM model the bridge serves (default: sonnet_bridge.py's "
                             "vertex_ai/claude-sonnet-4-6), e.g. vertex_ai/zai-org/glm-5-maas.")
    parser.add_argument("--bridge-vertex-location", default=None,
                        help="Vertex region for --bridge-model (bridge default us-east5).")
    parser.add_argument("--bridge-extra-body", type=json.loads, default=None,
                        help="JSON object sent as extra_body on every bridge call; config "
                             "files give it as a mapping.")
    parser.add_argument(
        "--evidence-source", default="web", choices=["web", "pubmed_s2", "s2"],
        help="'web' (default): Codex's native web_search (routed through sonnet_bridge.py to "
             "Anthropic's server-executed search). "
             "'pubmed_s2': web_search disabled; the system prompt instructs Codex to call "
             "ProClaim's own PubMed+S2 search backend via its shell tool instead (same "
             "functions react_baseline.py/pubmed_s2_mcp.py use). Also enables network access "
             "in the workspace-write sandbox (blocked by default), required for the PubMed/S2 "
             "HTTP calls to actually reach the network. 's2': as 'pubmed_s2' with Semantic "
             "Scholar only (for subsets outside PubMed's index). Output written to a separate "
             "{dataset}_{source}.jsonl so it never collides with the web-mode run.",
    )
    # Pre-parse --config so we can apply YAML defaults before the full parse.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config) as fh:
            cfg = yaml.safe_load(fh) or {}
        parser.set_defaults(**cfg)

    args = parser.parse_args()
    if not args.dataset:
        parser.error("--dataset is required (pass it on the command line or set "
                     "`dataset:` in the --config file)")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
