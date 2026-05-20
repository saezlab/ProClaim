#!/usr/bin/env python3
"""Merge ConnectomeDB chunk outputs and summarize timing reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


CHUNK_CSV_PATTERN = re.compile(r"results_chunk(\d+)\.csv$")
CHUNK_TIMING_PATTERN = re.compile(r"results_chunk(\d+)_timing\.json$")


def find_chunk_csvs(results_dir: Path) -> list[Path]:
    return sorted(
        path for path in results_dir.glob("results_chunk*.csv")
        if CHUNK_CSV_PATTERN.fullmatch(path.name)
    )


def find_chunk_timing_files(results_dir: Path) -> list[Path]:
    return sorted(
        path for path in results_dir.glob("results_chunk*_timing.json")
        if CHUNK_TIMING_PATTERN.fullmatch(path.name)
    )


def merge_chunk_csvs(chunk_paths: list[Path], output_path: Path) -> int:
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk CSVs found for {output_path.parent}")

    fieldnames: list[str] | None = None
    merged_rows = 0
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")

    with open(temp_path, "w", newline="") as f_out:
        writer = None
        for chunk_path in chunk_paths:
            with open(chunk_path, "r", newline="") as f_in:
                reader = csv.DictReader(f_in)
                if reader.fieldnames is None:
                    continue
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                    writer.writeheader()
                elif reader.fieldnames != fieldnames:
                    raise ValueError(f"Mismatched CSV columns in {chunk_path}")

                for row in reader:
                    writer.writerow(row)
                    merged_rows += 1

    if fieldnames is None:
        raise ValueError(f"Chunk CSVs in {output_path.parent} are empty")

    temp_path.replace(output_path)
    return merged_rows


def summarize_timing_reports(timing_paths: list[Path], summary_path: Path | None = None) -> dict[str, Any]:
    task_reports = []
    for timing_path in timing_paths:
        with open(timing_path, "r") as f:
            task_reports.append(json.load(f))

    task_reports.sort(key=lambda report: report.get("output_csv", ""))
    total_scheduled_jobs = sum(
        int(report.get("scheduled_jobs", report.get("pending_jobs", 0) + report.get("completed_jobs", 0)))
        for report in task_reports
    )
    total_completed_jobs = sum(int(report.get("completed_jobs", 0)) for report in task_reports)
    total_pending_jobs = sum(int(report.get("pending_jobs", 0)) for report in task_reports)
    max_task_wall = max((float(report.get("total_wall_seconds", 0.0)) for report in task_reports), default=0.0)
    sum_task_wall = sum(float(report.get("total_wall_seconds", 0.0)) for report in task_reports)

    worker_entries = []
    for report in task_reports:
        output_csv = report.get("output_csv", "")
        for worker_stat in report.get("worker_stats", []):
            worker_entries.append({
                "output_csv": output_csv,
                **worker_stat,
            })

    summary = {
        "results_dir": str(summary_path.parent if summary_path else timing_paths[0].parent if timing_paths else ""),
        "task_reports_found": len(task_reports),
        "scheduled_jobs": total_scheduled_jobs,
        "completed_jobs": total_completed_jobs,
        "pending_jobs": total_pending_jobs,
        "max_task_wall_seconds": round(max_task_wall, 2),
        "sum_task_wall_seconds": round(sum_task_wall, 2),
        "task_reports": task_reports,
        "worker_reports": worker_entries,
    }

    if summary_path is not None:
        summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def print_summary(chunk_paths: list[Path], timing_paths: list[Path], summary: dict[str, Any], output_path: Path, summary_path: Path) -> None:
    print(f"Chunk CSVs:      {len(chunk_paths)}")
    print(f"Timing reports:  {len(timing_paths)}")
    if output_path.exists():
        print(f"Merged CSV:      {output_path}")
    else:
        print(f"Merged CSV:      {output_path} (not written)")
    if timing_paths:
        print(f"Timing summary:  {summary_path}")
        print(f"Scheduled jobs:  {summary.get('scheduled_jobs', 0)}")
        print(f"Completed jobs:  {summary.get('completed_jobs', 0)}")
        print(f"Pending jobs:    {summary.get('pending_jobs', 0)}")
        print(f"Max task wall:   {summary.get('max_task_wall_seconds', 0.0):.2f}s")
        print(f"Sum task wall:   {summary.get('sum_task_wall_seconds', 0.0):.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge ConnectomeDB chunk outputs and summarize timing reports.")
    parser.add_argument("--results-dir", required=True, help="Path to results/connectomedb_eval_<RUN_TAG>")
    parser.add_argument("--output-csv", default="results.csv", help="Merged CSV name inside the results directory")
    parser.add_argument(
        "--timing-summary",
        default="timing_summary.json",
        help="Timing summary JSON name inside the results directory",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Only summarize timing reports and print status; do not rewrite results.csv.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = results_dir / args.output_csv
    summary_path = results_dir / args.timing_summary

    chunk_paths = find_chunk_csvs(results_dir)
    timing_paths = find_chunk_timing_files(results_dir)

    if not args.skip_merge:
        merged_rows = merge_chunk_csvs(chunk_paths, output_path)
        print(f"Merged {len(chunk_paths)} chunk CSVs ({merged_rows} rows) -> {output_path}")
    elif not output_path.exists():
        print(f"Merged CSV not present yet: {output_path}")

    summary = summarize_timing_reports(timing_paths, summary_path if timing_paths else None)
    print_summary(chunk_paths, timing_paths, summary, output_path, summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())