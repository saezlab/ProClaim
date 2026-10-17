"""Extract ProClaim's first subclaim list per claim, for the ReAct (with decomposed claims) arm.

ProClaim writes its subclaims in code (``state.subclaims = [...]``) and may rewrite them after
searching, so the final list in ``evidence_state.json`` can carry what its retrieval found. This
script takes the *first* assignment in each claim's ``workspace/execution_log.py``, which every
accepted run writes before its first search.

One output file per dataset and ProClaim run, keyed by the dataset CSV's ``claim_id``
(``_flip`` suffix for flipped SIGNOR rows); the ReAct run with seed S reads {dataset}_seed{S}.json:

    results/analysis/proclaim_subclaims/{dataset}_seed{seed}.json

    python scripts/analysis/extract_proclaim_subclaims.py \
        --dataset extri --seed 100 --run-dir results/extri_eval_<run-tag>
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
OUT = RES / "analysis/proclaim_subclaims"

_FIRST = re.compile(r"state\.subclaims\s*=\s*(\[.*?\])\s*\n", re.S)


def _norm(text: str) -> str:
    return " ".join(str(text).split()).rstrip(".").lower()


def _result_rows(run_dir: Path) -> list[dict]:
    merged = run_dir / "results.csv"
    files = [merged] if merged.exists() else sorted(run_dir.glob("results_chunk*.csv"))
    return [r for f in files for r in csv.DictReader(open(f, newline=""))]


def _first_subclaims(claim_dir: Path) -> list[str] | None:
    log = claim_dir / "workspace/execution_log.py"
    if not log.exists():
        return None
    m = _FIRST.search(log.read_text())
    if not m:
        return None
    try:
        value = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None
    return [str(s) for s in value] if isinstance(value, list) else None


def _claim_ids(dataset: str) -> dict[str, str]:
    ids = {}
    for r in csv.DictReader(open(REPO / "datasets" / f"{dataset}.csv", newline="")):
        flip = str(r.get("flip", "")).strip().lower() in ("true", "1")
        ids[_norm(r["claim"])] = r["id"] + ("_flip" if flip else "")
    return ids


def extract(dataset: str, seed: int, run_dir: Path) -> dict:
    ids = _claim_ids(dataset)
    out: dict[str, list[str]] = {}
    missing_log = 0
    for row in _result_rows(run_dir):
        cid = ids.get(_norm(row["Claim_String"]))
        if cid is None or cid in out:
            continue  # claim since removed from the CSV, or a duplicate row
        rel = Path(row["Output_Directory"])
        rel = Path(*rel.parts[rel.parts.index(run_dir.name) + 1:]) if run_dir.name in rel.parts else rel
        subclaims = _first_subclaims(run_dir / rel)
        if subclaims is None:
            missing_log += 1
            continue
        out[cid] = subclaims
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{dataset}_seed{seed}.json").write_text(json.dumps(out, indent=1))
    n = [len(v) for v in out.values()]
    print(f"{dataset} seed{seed}: {len(out)}/{len(ids)} claims, mean {sum(n)/max(len(n),1):.2f} "
          f"subclaims, no first list {missing_log}, not in run {len(ids) - len(out) - missing_log}  "
          f"<- {run_dir.name}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Dataset stem, e.g. signor or extri.")
    p.add_argument("--seed", type=int, required=True, help="Seed of the ReAct run that will read the output.")
    p.add_argument("--run-dir", type=Path, required=True, help="ProClaim run directory (holds results.csv).")
    args = p.parse_args()
    extract(args.dataset, args.seed, args.run_dir)


if __name__ == "__main__":
    main()
