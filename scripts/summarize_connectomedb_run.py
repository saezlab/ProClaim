#!/usr/bin/env python3
"""Summarize ConnectomeDB batch outputs and timing reports for one run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


def count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        row_count = sum(1 for _ in reader)
    return max(row_count - 1, 0)


def load_timing_reports(results_dir: Path) -> list[dict[str, Any]]:
    summary_name = "run_timing_summary.json"
    reports = []
    for timing_path in sorted(results_dir.glob("*_timing.json")):
        if timing_path.name == summary_name:
            continue
        try:
            report = json.loads(timing_path.read_text())
        except Exception as exc:
            reports.append(
                {
                    "report_path": str(timing_path),
                    "load_error": str(exc),
                    "completed_jobs": 0,
                    "total_wall_seconds": 0.0,
                    "merged_rows": 0,
                    "worker_stats": [],
                    "worker_shards": False,
                }
            )
            continue
        report["report_path"] = str(timing_path)
        reports.append(report)
    return reports


def build_summary(results_dir: Path) -> dict[str, Any]:
    final_csv = results_dir / "results.csv"
    chunk_csvs = sorted(str(path) for path in results_dir.glob("results_chunk*.csv"))
    shard_csvs = sorted(str(path) for path in results_dir.glob("results*.worker*.csv"))
    reports = load_timing_reports(results_dir)

    completed_jobs = sum(int(report.get("completed_jobs", 0)) for report in reports)
    merged_rows = sum(int(report.get("merged_rows", 0)) for report in reports)
    task_wall_seconds = [float(report.get("total_wall_seconds", 0.0)) for report in reports]
    all_worker_stats = [
        worker_stat
        for report in reports
        for worker_stat in report.get("worker_stats", [])
    ]

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results_dir": str(results_dir),
        "final_results_csv": str(final_csv),
        "final_results_rows": count_csv_rows(final_csv),
        "chunk_csvs": chunk_csvs,
        "chunk_csv_count": len(chunk_csvs),
        "worker_shard_csvs": shard_csvs,
        "worker_shard_csv_count": len(shard_csvs),
        "task_report_count": len(reports),
        "completed_jobs": completed_jobs,
        "merged_rows_reported": merged_rows,
        "sum_task_wall_seconds": round(sum(task_wall_seconds), 2),
        "max_task_wall_seconds": round(max(task_wall_seconds), 2) if task_wall_seconds else 0.0,
        "avg_task_wall_seconds": round(sum(task_wall_seconds) / len(task_wall_seconds), 2) if task_wall_seconds else 0.0,
        "worker_count_total": len(all_worker_stats),
        "worker_job_seconds_total": round(
            sum(float(worker_stat.get("job_seconds_total", 0.0)) for worker_stat in all_worker_stats),
            2,
        ),
        "worker_wall_seconds_total": round(
            sum(float(worker_stat.get("worker_wall_seconds", 0.0)) for worker_stat in all_worker_stats),
            2,
        ),
        "reports": reports,
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("Timing summary")
    print(f"  Results dir:          {summary['results_dir']}")
    print(f"  Final results rows:   {summary['final_results_rows']}")
    print(f"  Chunk CSV count:      {summary['chunk_csv_count']}")
    print(f"  Worker shard count:   {summary['worker_shard_csv_count']}")
    print(f"  Task report count:    {summary['task_report_count']}")
    print(f"  Completed jobs:       {summary['completed_jobs']}")
    print(f"  Max task wall time:   {summary['max_task_wall_seconds']} s")
    print(f"  Avg task wall time:   {summary['avg_task_wall_seconds']} s")
    print(f"  Sum task wall time:   {summary['sum_task_wall_seconds']} s")

    report_errors = [report for report in summary["reports"] if report.get("load_error")]
    if report_errors:
        print(f"  Timing load errors:   {len(report_errors)}")
        for report in report_errors[:3]:
            print(f"    {report['report_path']}: {report['load_error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ConnectomeDB run timing outputs.")
    parser.add_argument("--results-dir", required=True, help="Run results directory to summarize.")
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Do not fail when timing reports are missing; print current status instead.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional output path for the summary JSON (default: <results-dir>/run_timing_summary.json).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        if args.status_only:
            print(f"Timing summary unavailable: results dir not found: {results_dir}")
            return 0
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        return 1

    summary = build_summary(results_dir)
    summary_path = Path(args.summary_path) if args.summary_path else results_dir / "run_timing_summary.json"

    if summary["task_report_count"] == 0 and args.status_only:
        print("Timing summary unavailable: no *_timing.json files found yet")
        return 0
    if summary["task_report_count"] == 0:
        print("ERROR: no timing reports found", file=sys.stderr)
        return 1

    summary_path.write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    print(f"  Summary JSON:         {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
