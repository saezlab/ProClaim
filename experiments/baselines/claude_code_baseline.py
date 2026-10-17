"""Claude Code as a black-box agentic baseline, via ``claude_agent_sdk.query``.

Uses Claude Code's own default tools and backbone — no ``evidence_api``, no
prompt tuning beyond the task statement — so it benchmarks the off-the-shelf
harness rather than attributing performance to ProClaim's components.
``CLAUDE_CONFIG_DIR`` and ``cwd`` are redirected to a scratch dir for the run.

Usage::

    uv run python experiments/baselines/claude_code_baseline.py --dataset signor
    uv run python experiments/baselines/claude_code_baseline.py --dataset signor \\
        --config experiments/configs/claude_code_pubmed_s2_iclr_config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tempfile
import time
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
    PUBMED_S2_TOOL_HINT_MCP,
    S2_TOOL_HINT_MCP,
)

from baselines.shared.claim_loader import load_claim_csv
from baselines.shared.evaluate import EvaluationHarness
from baselines.shared.label_utils import get_label_config, normalize_label
from baselines.shared.logging_utils import write_claim_log
from baselines.shared.verdict import BaselineResult, extract_json

from claude_agent_sdk import (
    query, ClaudeAgentOptions, AssistantMessage, ResultMessage,
    TextBlock, ToolUseBlock, ServerToolUseBlock,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASETS_DIR = Path("datasets")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "baselines" / "claude_code"

# MCP server for --evidence-source pubmed_s2 (see pubmed_s2_mcp.py).
# Launched by script path, not `-m baselines.pubmed_s2_mcp`: the SDK's MCP
# client ignores a stdio server's "cwd", so `-m` can't resolve the package.
PUBMED_S2_MCP_SERVERS = {
    "pubmed-s2": {
        "command": sys.executable,
        "args": [str(PROJECT_ROOT / "experiments" / "baselines" / "pubmed_s2_mcp.py")],
    },
}
PUBMED_S2_ALLOWED_TOOLS = [
    "mcp__pubmed-s2__search_pubmed",
    "mcp__pubmed-s2__search_semantic_scholar",
]
PUBMED_S2_DISALLOWED_TOOLS = ["WebSearch", "WebFetch"]
# --evidence-source s2: the same MCP server with search_pubmed withheld. PubMed
# indexes none of the nuclear-physics or astronomy literature (NUBASE, NASA), so
# offering it there would spend turns on a source that cannot answer.
S2_ALLOWED_TOOLS = ["mcp__pubmed-s2__search_semantic_scholar"]

LC = get_label_config()


def build_system_prompt(evidence_source: str) -> str:
    tool_hint = {
        "pubmed_s2": PUBMED_S2_TOOL_HINT_MCP,
        "s2": S2_TOOL_HINT_MCP,
    }.get(evidence_source, "")
    return CLAUDE_CODE_BASELINE_SYSTEM_PROMPT.format(
        verdict_names=", ".join(LC.verdict_names()),
        verdict_definitions=LC.verdict_prompt_block(),
        label_options=" | ".join(f'"{n}"' for n in LC.verdict_names()),
        tool_hint=tool_hint,
    )


class ClaudeCodeBaseline:
    """Verify claims with the real Claude Code agent, driven as a black box."""

    name = "claude_code"

    def __init__(
        self,
        *,
        dataset: str = "",
        model: str | None = None,
        evidence_source: str = "web",
        max_turns: int = 25,
        max_budget_usd: float = 0.75,
        scratch_dir: Path | str | None = None,
        vertex: bool = False,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.vertex = vertex
        self.evidence_source = evidence_source
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd
        self.log_dir: Path | None = None

        scratch = Path(scratch_dir) if scratch_dir else Path(tempfile.mkdtemp(prefix=f"{self.name}_"))
        self.config_dir = scratch / "config"
        self.workdir = scratch / "workdir"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.workdir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Isolated Claude Code session dir (never touches the real ~/.claude): %s",
            self.config_dir,
        )

    def system_prompt(self, claim_id: str) -> str:
        """System prompt for one claim; per-claim so subclasses can vary it."""
        del claim_id
        return build_system_prompt(self.evidence_source)

    def _vertex_env(self) -> dict[str, str]:
        """Claude Code's own Vertex AI switches, from the repo's VERTEXAI_PROJECT.

        Region us-east5, as for the other ICLR baselines (VERTEXAI_LOCATION in
        .env is "global", which Claude Code accepts but prompt caching misses).
        """
        if not self.vertex:
            return {}
        env = {}
        # .env gives the service-account file relative to the repo root, but the
        # CLI runs in the scratch workdir, where a relative path does not resolve.
        creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds and not Path(creds).is_absolute():
            env["GOOGLE_APPLICATION_CREDENTIALS"] = str((PROJECT_ROOT / creds).resolve())
        return {
            **env,
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_PROJECT_ID": os.environ["VERTEXAI_PROJECT"],
            "CLOUD_ML_REGION": os.environ.get("CLAUDE_CODE_VERTEX_REGION", "us-east5"),
        }

    def _options(self, claim_id: str) -> ClaudeAgentOptions:
        if self.evidence_source in ("pubmed_s2", "s2"):
            # No permission_mode: "bypassPermissions" would make Claude Code
            # ignore allowed_tools/disallowed_tools entirely.
            extra_options = dict(
                mcp_servers=PUBMED_S2_MCP_SERVERS,
                allowed_tools=(
                    PUBMED_S2_ALLOWED_TOOLS
                    if self.evidence_source == "pubmed_s2"
                    else S2_ALLOWED_TOOLS
                ),
                disallowed_tools=PUBMED_S2_DISALLOWED_TOOLS,
            )
        else:
            extra_options = dict(permission_mode="bypassPermissions")
        return ClaudeAgentOptions(
            system_prompt=self.system_prompt(claim_id),
            cwd=str(self.workdir),
            env={**os.environ, "CLAUDE_CONFIG_DIR": str(self.config_dir), **self._vertex_env()},
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
            model=self.model,
            **extra_options,
        )

    async def averify(
        self,
        claim_id: str,
        claim: str,
        gold_label: str,
        *,
        context: dict | None = None,
    ) -> BaselineResult:
        del context
        user_prompt = CLAUDE_CODE_BASELINE_USER_PROMPT.format(claim=claim)

        t0 = time.monotonic()
        final_text = ""
        model_used = ""
        num_turns = 0
        cost = 0.0
        tool_calls: list[str] = []
        error_msg = ""

        try:
            async for message in query(prompt=user_prompt, options=self._options(claim_id)):
                if isinstance(message, AssistantMessage):
                    model_used = message.model or model_used
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            final_text = block.text
                        elif isinstance(block, (ToolUseBlock, ServerToolUseBlock)):
                            tool_calls.append(block.name)
                elif isinstance(message, ResultMessage):
                    num_turns = message.num_turns or 0
                    cost = message.total_cost_usd or 0.0
                    if message.result:
                        final_text = message.result
        except Exception as exc:
            # A cap being hit arrives as the CLI exiting 1, not a partial
            # result, so it's indistinguishable from any other crash here.
            error_msg = str(exc) or f"{type(exc).__name__} (no message)"

        elapsed = time.monotonic() - t0
        parsed = extract_json(final_text)
        # Nothing parseable after a failure is an ERROR, not UNCERTAIN.
        failed = bool(error_msg) and not parsed.get("label")
        predicted = normalize_label(parsed.get("label")) if parsed.get("label") else "UNCERTAIN"
        reasoning = parsed.get("reasoning", "") or error_msg
        if failed:
            predicted = "ERROR"
            logger.error(
                "%s produced no verdict after $%.3f / %d turns: %s",
                claim_id, cost, num_turns, error_msg,
            )

        write_claim_log(
            self.log_dir,
            claim_id,
            [
                (f"claim_id: {claim_id}", ""),
                ("claim", claim),
                (f"evidence_source: {self.evidence_source}", ""),
                (f"model: {model_used}  turns: {num_turns}  cost_usd: {cost:.4f}", ""),
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
            reasoning=reasoning,
            evidence=tool_calls,
            cost_usd=cost,
            latency_seconds=elapsed,
            baseline_name=self.name,
            model=model_used,
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

    baseline = ClaudeCodeBaseline(
        dataset=args.dataset,
        model=args.model,
        evidence_source=args.evidence_source,
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
        scratch_dir=args.scratch_dir,
        vertex=args.vertex,
    )
    harness = EvaluationHarness(baseline, dataset_name=args.dataset)

    output_dir = Path(args.output_dir)
    stem = f"{args.dataset}_{args.evidence_source}"
    output_path = output_dir / f"{stem}.jsonl"
    # Mirrors run_baselines_datasets.py:409. Without this the per-claim logs
    # built below are silently discarded (write_claim_log no-ops on None).
    baseline.log_dir = output_dir / f"{stem}_logs"

    t_start = time.monotonic()
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
             "experiments/configs/claude_code_*_config.yaml). Keys are the "
             "flags' long names with dashes as underscores, e.g. "
             "`max_budget_usd`. Any flag passed on the command line overrides "
             "the config.",
    )
    # Validated after parsing, not via required=True, so --config can supply it.
    # No `choices=`: the loader is dataset-agnostic (it reads {dataset}.csv from
    # --datasets-dir), and the system prompt is conditioned only on
    # --evidence-source, never on the dataset. Restricting the list here was the
    # thing stopping the other ProClaim-eval subsets from reusing this driver.
    parser.add_argument("--dataset", default=None,
                        help="Dataset stem; {dataset}.csv must exist in --datasets-dir.")
    parser.add_argument("--datasets-dir", default=str(DEFAULT_DATASETS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scratch-dir", default=None,
                         help="Scratch dir for isolated Claude Code session state. "
                              "Defaults to a fresh temp dir (auto-cleaned by the OS).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--max-budget-usd", type=float, default=0.75)
    parser.add_argument(
        "--model", default=None,
        help="Backbone model (e.g. claude-opus-4-8). Unset by default.",
    )
    parser.add_argument(
        "--vertex", action="store_true",
        help="Serve the model through Vertex AI (project from VERTEXAI_PROJECT, region "
             "us-east5 unless CLAUDE_CODE_VERTEX_REGION is set) instead of the Anthropic API.",
    )
    parser.add_argument(
        "--evidence-source", default="web", choices=["web", "pubmed_s2", "s2"],
        help="'web' (default): Claude Code's native WebSearch/WebFetch. 'pubmed_s2': "
             "those disabled, replaced by an MCP server exposing ProClaim's PubMed+S2 "
             "backend. Written to a separate {dataset}_{source}.jsonl.",
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
