#!/usr/bin/env python3
"""
Test whether web search queries from React+WebSearch logs can retrieve
ConnectomeDB-specific information, indicating potential data leakage.

Risk classification:
  HIGH   - query contains "connectome" or uses GENE1_GENE2 underscore format
  MEDIUM - query searches a specific gene pair in databases
  LOW    - general biology/mechanism query

Usage:
  uv run python experiments/test_search_leakage.py \
      --logs-dir /hps/nobackup/saezrodriguez/ail/workspace/grn-llm-correct/results/baselines/react/web/anthropic--claude-sonnet-4-6/connectomedb_seed100_logs \
      --dataset /hps/nobackup/saezrodriguez/ail/workspace/grn-llm-correct/results/baselines/random/connectomedb_seed100.jsonl \
      --risk-level medium \
      --output results/leakage_test.jsonl
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


# ── Risk classification ──────────────────────────────────────────────────────

# Patterns that strongly suggest the query is probing ConnectomeDB specifically
_HIGH_RISK_PATTERNS = [
    re.compile(r"connectome", re.IGNORECASE),
    re.compile(r"CLRIA", re.IGNORECASE),
    # underscore-joined gene pair: uppercase letters/digits only
    re.compile(r"\b[A-Z][A-Z0-9]+_[A-Z][A-Z0-9]+\b"),
    re.compile(r"connectomedb", re.IGNORECASE),
]

# Patterns for medium risk: specific gene pair + database name
_MEDIUM_RISK_PATTERNS = [
    re.compile(r"(STRING|BioGRID|IntAct|CellChat|NicheNet|CellPhoneDB|OmniPath|CellTalkDB|HPMR|IUPHAR)", re.IGNORECASE),
]

# ConnectomeDB leakage indicators in search result text
_LEAKAGE_INDICATORS = [
    re.compile(r"CDB25:\d+", re.IGNORECASE),
    re.compile(r"connectomedb", re.IGNORECASE),
    re.compile(r"connectome\s+db", re.IGNORECASE),
    re.compile(r"is a ligand that directly interacts with receptor", re.IGNORECASE),
    # Official ConnectomeDB website and GitHub repo
    re.compile(r"connectomedb\.org", re.IGNORECASE),
    re.compile(r"github\.com/bioinfo-YCU/ConnectomeDB", re.IGNORECASE),
    # Broader GitHub connectome pattern
    re.compile(r"github\.com.*connectome", re.IGNORECASE),
    re.compile(r"scverse.*connectome|connectome.*scverse", re.IGNORECASE),
]


def classify_risk(query: str) -> str:
    for pat in _HIGH_RISK_PATTERNS:
        if pat.search(query):
            return "HIGH"
    for pat in _MEDIUM_RISK_PATTERNS:
        if pat.search(query):
            return "MEDIUM"
    return "LOW"


# ── Log parsing ──────────────────────────────────────────────────────────────

def parse_logs(logs_dir: Path) -> list[dict]:
    """Extract (claim_id, query, risk_level) from all log files."""
    records = []
    for log_file in sorted(logs_dir.glob("*.log")):
        claim_id = log_file.stem  # filename without extension
        content = log_file.read_text()

        queries = re.findall(r"Q: (.+)", content)
        for q in queries:
            q = q.strip()
            records.append({
                "claim_id": claim_id,
                "query": q,
                "risk": classify_risk(q),
            })
    return records


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_gold_labels(dataset_path: Path) -> dict[str, str]:
    """Return {gene_pair: gold_label} from a JSONL dataset file."""
    labels = {}
    with open(dataset_path) as f:
        for line in f:
            d = json.loads(line)
            claim = d.get("claim", "")
            m = re.match(r"(\S+) is a ligand that directly interacts with receptor (\S+)", claim)
            if m:
                pair = f"{m.group(1)} {m.group(2)}"
                labels[pair] = d["gold_label"]
    return labels


# ── Serper search ────────────────────────────────────────────────────────────

def serper_search(query: str, api_key: str, k: int = 5) -> dict:
    """Run a single Serper search and return the raw JSON response."""
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    params = {"q": query, "num": k, "gl": "us", "hl": "en"}
    resp = requests.post(
        "https://google.serper.dev/search",
        headers=headers,
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def extract_text(serper_resp: dict) -> str:
    """Flatten Serper response to plain text for leakage scanning."""
    parts = []
    if ab := serper_resp.get("answerBox"):
        for f in ("answer", "snippet", "snippetHighlighted"):
            if v := ab.get(f):
                parts.append(str(v))
    if kg := serper_resp.get("knowledgeGraph"):
        if d := kg.get("description"):
            parts.append(d)
    for item in serper_resp.get("organic", []):
        for f in ("title", "snippet", "link"):
            if v := item.get(f):
                parts.append(str(v))
    return " ".join(parts)


# ── Leakage check ────────────────────────────────────────────────────────────

def check_leakage(text: str) -> list[str]:
    """Return list of matched leakage indicator descriptions."""
    matched = []
    for pat in _LEAKAGE_INDICATORS:
        m = pat.search(text)
        if m:
            matched.append(f"{pat.pattern!r} matched: ...{text[max(0,m.start()-30):m.end()+30]!r}...")
    return matched


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Test ConnectomeDB data leakage via web search")
    parser.add_argument(
        "--logs-dir",
        required=True,
        type=Path,
        help="Directory containing *.log files from react+web baseline",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="JSONL file with gold labels (connectomedb_seed100.jsonl)",
    )
    parser.add_argument(
        "--risk-level",
        choices=["all", "high", "medium"],
        default="high",
        help="Minimum risk level to test (default: high)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/leakage_test.jsonl"),
        help="Output JSONL file for results",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between Serper requests (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and classify queries only; do not run web searches",
    )
    args = parser.parse_args()

    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key and not args.dry_run:
        raise SystemExit("SERPER_API_KEY not set. Export it or use --dry-run.")

    print(f"Parsing logs from: {args.logs_dir}")
    all_records = parse_logs(args.logs_dir)
    print(f"  Total queries found: {len(all_records)}")

    # Count by risk
    from collections import Counter
    risk_counts = Counter(r["risk"] for r in all_records)
    for level in ("HIGH", "MEDIUM", "LOW"):
        print(f"  {level}: {risk_counts[level]}")

    # Filter by risk level
    threshold = {"all": {"HIGH", "MEDIUM", "LOW"}, "high": {"HIGH"}, "medium": {"HIGH", "MEDIUM"}}[args.risk_level]
    to_test = [r for r in all_records if r["risk"] in threshold]

    # Deduplicate queries (same query may appear across multiple logs)
    seen_queries: dict[str, list[str]] = {}  # query -> [claim_ids]
    for r in to_test:
        seen_queries.setdefault(r["query"], []).append(r["claim_id"])

    print(f"\nUnique queries to test ({args.risk_level}+): {len(seen_queries)}")

    print(f"\nLoading gold labels from: {args.dataset}")
    gold_labels = load_gold_labels(args.dataset)
    print(f"  Gold labels loaded: {len(gold_labels)}")

    if args.dry_run:
        print("\n--- DRY RUN: queries that would be tested ---")
        for q, claim_ids in sorted(seen_queries.items(), key=lambda x: classify_risk(x[0])):
            risk = classify_risk(q)
            affected_labels = {gold_labels.get(cid, "?") for cid in claim_ids}
            print(f"  [{risk}] {q!r}")
            print(f"         claims: {claim_ids}")
            print(f"         gold labels exposed if leaked: {affected_labels}")
        return

    # Run actual searches
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    leakage_found = 0

    print(f"\nRunning {len(seen_queries)} searches...")
    for i, (query, claim_ids) in enumerate(seen_queries.items(), 1):
        risk = classify_risk(query)
        affected_labels = {gold_labels.get(cid, "?") for cid in claim_ids}
        print(f"[{i}/{len(seen_queries)}] [{risk}] {query!r}")

        try:
            resp = serper_search(query, api_key, k=5)
            result_text = extract_text(resp)
            leakage_hits = check_leakage(result_text)
            leaked = len(leakage_hits) > 0
            if leaked:
                leakage_found += 1
                print(f"  *** LEAKAGE DETECTED ***")
                for hit in leakage_hits:
                    print(f"    {hit}")

            record = {
                "query": query,
                "risk": risk,
                "claim_ids": claim_ids,
                "affected_gold_labels": sorted(affected_labels),
                "leaked": leaked,
                "leakage_hits": leakage_hits,
                "top_results": [
                    {"title": r.get("title", ""), "link": r.get("link", ""), "snippet": r.get("snippet", "")}
                    for r in resp.get("organic", [])[:5]
                ],
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            record = {
                "query": query,
                "risk": risk,
                "claim_ids": claim_ids,
                "affected_gold_labels": sorted(affected_labels),
                "leaked": None,
                "leakage_hits": [],
                "error": str(e),
                "top_results": [],
            }

        results.append(record)
        with open(args.output, "a") as f:
            f.write(json.dumps(record) + "\n")

        if i < len(seen_queries):
            time.sleep(args.delay)

    # Summary
    tested = len([r for r in results if r.get("leaked") is not None])
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Queries tested:   {tested}")
    print(f"  Leakage detected: {leakage_found}")
    print(f"  Output:           {args.output}")
    if leakage_found:
        print(f"\n  Leaking queries:")
        for r in results:
            if r.get("leaked"):
                print(f"    [{r['risk']}] {r['query']!r}")
                print(f"         gold labels: {r['affected_gold_labels']}")


if __name__ == "__main__":
    main()
