#!/usr/bin/env python3
"""Convert direct-eval CSV results into baseline-style JSONL.

This is intended for ProClaim-style CSV outputs such as:

    results/baselines/signor_direct_eval_20260427_221617/results.csv
    results/baselines/connectomedb_eval_20260424_171117/.../results.csv

and emits one JSON object per line using the shared ``BaselineResult`` schema,
matching the structure used by files such as:

    results/baselines/single_paper/anthropic--claude-sonnet-4-6/signor_seed100.jsonl
    results/baselines/single_paper/anthropic--claude-sonnet-4-6/connectomedb_seed100.jsonl

For supported datasets, the converter uses baseline-compatible claim IDs:

- SIGNOR original claim: ``SIGNOR-123``
- SIGNOR flipped claim: ``SIGNOR-123_flip``
- ConnectomeDB claim: ``CDB25:0003056``

By default only ``Repetition == 1`` rows are exported so the output contains
one row per claim variant, like the existing baseline JSONL files.

Usage
-----
uv run python scripts/analysis/convert_direct_eval_csv_to_jsonl.py \
    results/baselines/signor_direct_eval_20260427_221617/results.csv \
    --output results/baselines/signor_direct_eval_20260427_221617/signor_seed100.jsonl

uv run python scripts/analysis/convert_direct_eval_csv_to_jsonl.py \
    results/baselines/connectomedb_eval_20260424_171117/results.csv \
    --output results/baselines/connectomedb_eval_20260424_171117/connectomedb_seed100.jsonl
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from experiments.baselines.shared.evaluate import EvaluationHarness
from experiments.baselines.shared.label_utils import normalize_label
from experiments.baselines.shared.verdict import BaselineResult


def _schema_kind(fieldnames: list[str] | None) -> str:
    field_set = set(fieldnames or [])
    if "SIGNOR_ID" in field_set:
        return "signor"
    if "CDB_ID" in field_set:
        return "connectomedb"
    raise ValueError("Unsupported CSV schema: expected SIGNOR_ID or CDB_ID columns.")


def _default_metadata(schema_kind: str) -> tuple[str, str]:
    if schema_kind == "signor":
        return "signor_direct_eval", "SIGNOR"
    return "connectomedb_direct_eval", "ConnectomeDB"


def _signor_claim_id(row: dict[str, str]) -> str:
    claim_id = row["SIGNOR_ID"].strip()
    is_flipped = row["Is_Flipped"].strip().lower() in {"true", "1", "yes"}
    if is_flipped:
        return f"{claim_id}_flip"
    return claim_id


def _connectomedb_claim_id(row: dict[str, str]) -> str:
    return row["CDB_ID"].strip()


def _result_from_row(
    row: dict[str, str],
    *,
    claim_id: str,
    gold_label: str,
    baseline_name: str,
    model: str,
    dataset: str,
) -> BaselineResult:
    return BaselineResult(
        claim_id=claim_id,
        claim=(row.get("Claim_String") or "").strip(),
        gold_label=normalize_label(gold_label),
        predicted_label=normalize_label(row.get("Agent_Verdict") or ""),
        confidence=float(row.get("Agent_Confidence") or 0.0),
        reasoning=(row.get("Reasoning_Snippet") or "").strip(),
        evidence=[],
        input_tokens=int(float(row.get("Total_Input_Tokens") or 0)),
        output_tokens=int(float(row.get("Output_Tokens") or 0)),
        cost_usd=float(row.get("Cost_Estimate") or 0.0),
        baseline_name=baseline_name,
        model=model,
        dataset=dataset,
    )


def convert_csv(
    csv_path: Path,
    *,
    repetition: str,
    baseline_name: str,
    model: str,
    dataset: str,
) -> list[BaselineResult]:
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        schema_kind = _schema_kind(reader.fieldnames)
        default_baseline_name, default_dataset = _default_metadata(schema_kind)
        baseline_name = baseline_name or default_baseline_name
        dataset = dataset or default_dataset

        if schema_kind == "signor":
            return _convert_signor_rows(
                reader,
                repetition=repetition,
                baseline_name=baseline_name,
                model=model,
                dataset=dataset,
            )

        return _convert_connectomedb_rows(
            reader,
            repetition=repetition,
            baseline_name=baseline_name,
            model=model,
            dataset=dataset,
        )


def _convert_signor_rows(
    reader: csv.DictReader,
    *,
    repetition: str,
    baseline_name: str,
    model: str,
    dataset: str,
) -> list[BaselineResult]:
    result_by_claim_id: dict[str, BaselineResult] = {}
    base_claim_order: list[str] = []
    seen_base_claim_ids: set[str] = set()
    seen_claim_ids: set[str] = set()

    required = {
        "SIGNOR_ID",
        "Flipped_Label",
        "Claim_String",
        "Is_Flipped",
        "Agent_Verdict",
    }
    missing = required.difference(reader.fieldnames or [])
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required SIGNOR columns: {missing_str}")

    for row in reader:
        row_repetition = (row.get("Repetition") or "1").strip()
        if repetition != "all" and row_repetition != repetition:
            continue

        claim_id = _signor_claim_id(row)
        base_claim_id = row["SIGNOR_ID"].strip()
        if claim_id in seen_claim_ids:
            raise ValueError(
                "Duplicate baseline-style claim_id encountered after filtering: "
                f"{claim_id}. Use --repetition to select a single repetition."
            )
        seen_claim_ids.add(claim_id)
        if base_claim_id not in seen_base_claim_ids:
            base_claim_order.append(base_claim_id)
            seen_base_claim_ids.add(base_claim_id)

        result_by_claim_id[claim_id] = _result_from_row(
            row,
            claim_id=claim_id,
            gold_label=row["Flipped_Label"],
            baseline_name=baseline_name,
            model=model,
            dataset=dataset,
        )

    ordered_results: list[BaselineResult] = []
    for base_claim_id in base_claim_order:
        for claim_id in (base_claim_id, f"{base_claim_id}_flip"):
            result = result_by_claim_id.get(claim_id)
            if result is not None:
                ordered_results.append(result)

    return ordered_results


def _convert_connectomedb_rows(
    reader: csv.DictReader,
    *,
    repetition: str,
    baseline_name: str,
    model: str,
    dataset: str,
) -> list[BaselineResult]:
    results: list[BaselineResult] = []
    seen_claim_ids: set[str] = set()

    required = {
        "CDB_ID",
        "Label",
        "Claim_String",
        "Agent_Verdict",
    }
    missing = required.difference(reader.fieldnames or [])
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required ConnectomeDB columns: {missing_str}")

    for row in reader:
        row_repetition = (row.get("Repetition") or "1").strip()
        if repetition != "all" and row_repetition != repetition:
            continue

        claim_id = _connectomedb_claim_id(row)
        if claim_id in seen_claim_ids:
            raise ValueError(
                "Duplicate ConnectomeDB claim_id encountered after filtering: "
                f"{claim_id}. Use --repetition to select a single repetition."
            )
        seen_claim_ids.add(claim_id)

        results.append(
            _result_from_row(
                row,
                claim_id=claim_id,
                gold_label=row["Label"],
                baseline_name=baseline_name,
                model=model,
                dataset=dataset,
            )
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert direct-eval CSV results into baseline-style JSONL."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the direct-eval CSV file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to <csv_dir>/<csv_stem>.jsonl.",
    )
    parser.add_argument(
        "--repetition",
        default="1",
        help="Which Repetition value to export. Use 'all' only if the filtered data remains unique by claim_id.",
    )
    parser.add_argument(
        "--baseline-name",
        default="",
        help="Value written into the baseline_name field. Defaults to a schema-specific name.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Value written into the model field.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Value written into the dataset field. Defaults to a schema-specific dataset name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.csv_path.with_suffix(".jsonl")

    results = convert_csv(
        args.csv_path,
        repetition=args.repetition,
        baseline_name=args.baseline_name,
        model=args.model,
        dataset=args.dataset,
    )
    EvaluationHarness.save(results, output_path)
    print(f"Converted {len(results)} rows to {output_path}")


if __name__ == "__main__":
    main()