"""
OpenScholar baseline — retrieval-augmented generation via the OpenScholar pipeline.

Frames each claim as a literature question, then routes it through OpenScholar
(backed by a configurable API model).  By default the model generates a verdict
using its parametric knowledge together with the claim_verdict task prompt, which
asks for SUPPORT / REFUTE / UNCERTAIN with a reasoning sentence.

If ``use_retrieval=True`` the wrapper also passes ``--ss_retriever --feedback``
to OpenScholar so it queries Semantic Scholar for supporting passages before
generating the final verdict (requires S2 network access; anonymous requests are
accepted but rate-limited).

Cost: 1–3 LLM calls per claim depending on retrieval/feedback flags.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from baselines.shared.label_utils import normalize_label
from baselines.shared.verdict import BaselineResult

logger = logging.getLogger(__name__)

# Resolve paths relative to *this* file:
#   baselines/open_scholar_baseline.py
#   ../../..  → grn-llm-correct/
#   ../../../../OpenScholar → OpenScholar root
_BASELINES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASELINES_DIR.parent.parent      # grn-llm-correct/
_WORKSPACE_ROOT = _PROJECT_ROOT.parent             # workspace/
_DEFAULT_OPEN_SCHOLAR_DIR = _WORKSPACE_ROOT / "OpenScholar"
_DEFAULT_ENV_FILE = _PROJECT_ROOT / ".env"


def _load_dotenv(env_file: Path) -> dict[str, str]:
    """Parse a .env file and return a dict of key/value pairs."""
    env: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


class OpenScholarBaseline:
    """Verify claims via the OpenScholar pipeline (subprocess, API-model backend).

    Parameters
    ----------
    model:
        LiteLLM / API model name as expected by OpenScholar's ``--model_name``.
    api:
        OpenScholar ``--api`` provider string (``"anthropic"``, ``"gemini"``,
        ``"together"``, etc.).
    top_n:
        Number of retrieved passages forwarded to the generator (``--top_n``).
    max_tokens:
        Maximum tokens for the final generation step.  ``None`` omits the flag
        so OpenScholar's own default (3000) is used.
    use_retrieval:
        If ``True``, add ``--ss_retriever --feedback`` so OpenScholar issues
        Semantic Scholar search queries and augments the context.
    open_scholar_dir:
        Path to the OpenScholar repository root.  Defaults to
        ``<workspace>/OpenScholar``.
    env_file:
        Path to a ``.env`` file containing ``ANTHROPIC_API_KEY`` etc.  Keys
        already present in the process environment take precedence.
    s2_api_key:
        Semantic Scholar API key.  Pass ``""`` for anonymous (rate-limited)
        access.  Defaults to the ``S2_API_KEY`` env var or empty string.
    subprocess_timeout:
        Seconds to wait for a single OpenScholar subprocess call.
    """

    name = "open_scholar"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api: str = "anthropic",
        top_n: int = 5,
        max_tokens: int | None = None,
        use_retrieval: bool = False,
        reranker: str | None = "OpenScholar/OpenScholar_Reranker",
        open_scholar_dir: str | Path | None = None,
        env_file: str | Path | None = None,
        s2_api_key: str | None = None,
        subprocess_timeout: int = 300,
    ) -> None:
        self.model = model
        self.api = api
        self.top_n = top_n
        self.max_tokens = max_tokens  # None → omit flag; OpenScholar default is 3000
        self.use_retrieval = use_retrieval
        self.reranker = reranker  # None → no reranking; required with --ss_retriever --feedback
        # Increase default timeout when retrieval is active (S2 + feedback can be slow)
        self.subprocess_timeout = subprocess_timeout if subprocess_timeout != 300 else (600 if use_retrieval else 300)

        self._open_scholar_dir = Path(open_scholar_dir or _DEFAULT_OPEN_SCHOLAR_DIR)
        self._python = str(self._open_scholar_dir / ".venv" / "bin" / "python")
        self._env_file = Path(env_file or _DEFAULT_ENV_FILE)

        # Resolve S2 API key: explicit arg > env var > empty (anonymous)
        if s2_api_key is not None:
            self._s2_api_key = s2_api_key
        else:
            self._s2_api_key = os.environ.get("S2_API_KEY", "")

        # Build the environment that subprocess calls will inherit
        dotenv = _load_dotenv(self._env_file)
        self._subprocess_env: dict[str, str] = {**dotenv, **os.environ}  # process env wins
        self._subprocess_env.setdefault("S2_API_KEY", self._s2_api_key)

    # ------------------------------------------------------------------
    # Public interface

    def verify(
        self,
        claim_id: str,
        claim: str,
        gold_label: str,
        context: dict | None = None,
    ) -> BaselineResult:
        """Run one claim through OpenScholar and return a BaselineResult.

        Parameters
        ----------
        context:
            Optional claim dict forwarded by ``EvaluationHarness``.  When the
            dict contains an ``"evidence"`` key (e.g. from the SIGNOR/ConnectomeDB
            CSVs), that snippet is used as a pre-retrieved context passage so that
            OpenScholar can render a grounded verdict without live S2 retrieval.
        """
        t0 = time.monotonic()

        # Pass the raw claim text.  OpenScholar's `claim_verdict` task already
        # prepends "Given a scientific claim and a set of retrieved reference
        # passages …" as its system instruction and uses "\nClaim: " as the
        # instance header, so the model receives:
        #   <system instruction>
        #   References: [0] … [1] …
        #   Claim: <claim text>
        # Wrapping the claim as a question ("What does the scientific evidence say
        # about …?") placed after the "Claim:" header produces a malformed
        # prompt and causes the model to fabricate reference citations.
        framed_input = claim

        # Build ctxs from pre-existing evidence when available
        ctxs: list[dict] = []
        use_contexts_flag = self.use_retrieval  # default follows retrieval flag
        if context and context.get("evidence"):
            evidence_text = context["evidence"]
            pmid = context.get("pmid", "")
            ctxs = [{"title": f"PMID:{pmid}" if pmid else "", "text": evidence_text}]
            use_contexts_flag = True  # we have evidence to ground the verdict

        with tempfile.TemporaryDirectory(prefix="os_baseline_") as tmpdir:
            input_file = Path(tmpdir) / "input.jsonl"
            output_file = Path(tmpdir) / "output.json"

            input_file.write_text(
                json.dumps({"input": framed_input, "ctxs": ctxs, "answer": ""}) + "\n"
            )

            cmd = self._build_cmd(input_file, output_file, use_contexts=use_contexts_flag)
            logger.debug("Running OpenScholar for claim %s: %s", claim_id, " ".join(cmd))

            proc = subprocess.run(
                cmd,
                cwd=str(self._open_scholar_dir),
                env=self._subprocess_env,
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout,
            )

            if proc.returncode != 0:
                stderr_tail = proc.stderr[-1000:] if proc.stderr else ""
                raise RuntimeError(
                    f"OpenScholar subprocess failed for claim {claim_id}.\n{stderr_tail}"
                )

            raw_data = json.loads(output_file.read_text())

        latency = time.monotonic() - t0
        item = raw_data["data"][0] if "data" in raw_data else raw_data[0]
        raw_output: str = item.get("output", "")
        cost_usd = float(item.get("total_cost", 0.0))

        verdict = self._parse_verdict(raw_output)

        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=normalize_label(verdict.get("label", "NEI")),
            confidence=0.0,
            reasoning=verdict.get("reasoning", raw_output[:500] if raw_output else ""),
            evidence=verdict.get("evidence", []),
            # OpenScholar reports cost in USD but not token counts
            input_tokens=0,
            output_tokens=0,
            cost_usd=cost_usd,
            latency_seconds=latency,
            baseline_name=self.name,
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Helpers

    def _build_cmd(
        self, input_file: Path, output_file: Path, use_contexts: bool = False
    ) -> list[str]:
        cmd = [
            self._python, "run.py",
            "--input_file", str(input_file),
            "--output_file", str(output_file),
            "--api", self.api,
            "--model_name", self.model,
            "--task_name", "claim_verdict",
            "--zero_shot",
            "--top_n", str(self.top_n),
        ]
        if self.max_tokens is not None:
            cmd += ["--max_tokens", str(self.max_tokens)]
        if use_contexts:
            cmd.append("--use_contexts")
        if self.use_retrieval:
            cmd += ["--ss_retriever", "--feedback"]
            if self.reranker:
                cmd += ["--ranking_ce", "--reranker", self.reranker]
        return cmd

    @staticmethod
    def _parse_verdict(raw_output: str) -> dict:
        """Parse the JSON verdict from OpenScholar output.

        Falls back to a NEI result rather than raising on malformed JSON.
        """
        text = raw_output.strip()
        # Strip markdown code fences that occasionally appear
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse OpenScholar verdict JSON: %r", text[:200])
            return {"label": "NEI", "reasoning": text[:500], "evidence": []}
