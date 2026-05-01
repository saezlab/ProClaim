"""Focused tests for shared baseline utilities."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


from baselines.shared.cost_tracker import CostTracker
from baselines.shared.logging_utils import write_claim_log
from baselines.shared.search_utils import format_s2_detailed_results
from baselines.shared.single_shot import run_single_shot_verdict


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, system: str, user: str, *, response_format: str = "json_object"):
        return self._text, 11, 7

    def complete_text(self, system: str, user: str):
        return self._text, 11, 7

    def parse_json(self, text: str):
        import json

        return json.loads(text)


def test_write_claim_log_writes_named_sections(tmp_path):
    write_claim_log(
        tmp_path,
        "claim-1",
        [("claim", "A activates B"), ("reasoning", "because evidence says so")],
    )

    contents = (tmp_path / "claim-1.log").read_text()
    assert "=== claim ===" in contents
    assert "A activates B" in contents
    assert "=== reasoning ===" in contents


def test_format_s2_detailed_results_includes_authors_and_citations():
    rendered = format_s2_detailed_results(
        [
            {
                "title": "Paper title",
                "year": 2024,
                "abstract": "Abstract text.",
                "authors": [{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}],
                "externalIds": {"PubMed": "12345"},
                "citationCount": 17,
            }
        ]
    )

    assert "Paper title (2024)" in rendered
    assert "A, B, C et al." in rendered
    assert "PMID: 12345" in rendered
    assert "Citations: 17" in rendered


def test_run_single_shot_verdict_tracks_and_parses_output():
    llm = _FakeLLM('{"label": "SUPPORT", "confidence": 0.8, "reasoning": "Strong evidence", "evidence": ["PMID1"]}')
    tracker = CostTracker(model="anthropic/claude-sonnet-4-6")

    verdict = run_single_shot_verdict(
        llm=llm,
        tracker=tracker,
        system_prompt="system",
        user_prompt="user",
    )

    assert verdict.predicted_label == "SUPPORT"
    assert verdict.confidence == 0.8
    assert verdict.reasoning == "Strong evidence"
    assert verdict.evidence == ["PMID1"]
    assert verdict.summary["input_tokens"] == 11
    assert verdict.summary["output_tokens"] == 7