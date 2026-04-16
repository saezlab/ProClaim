"""
Run SLM/LLM/MLP inference for the sufficiency classifier ablation.

Backends:
  slm  — Qwen via vLLM (reads llm_sufficiency_prompt.txt from ablation-dir)
  llm  — Claude Haiku via Anthropic API (same input)
  mlp  — Existing MLP classifier using pre-extracted numerical features
         (reads evidence_state.json from base-dir, writes to ablation-dir)

Usage:
    uv run python scripts/sufficiency_classifier/slm_llm_ablation/run_inference.py --backend slm --overwrite
    uv run python scripts/sufficiency_classifier/slm_llm_ablation/run_inference.py --backend llm --overwrite
    uv run python scripts/sufficiency_classifier/slm_llm_ablation/run_inference.py --backend mlp
    uv run python scripts/sufficiency_classifier/slm_llm_ablation/run_inference.py --backend slm --dry-run
"""

import json
import glob
import os
import re
import argparse
import time
import sys
from pathlib import Path

# ── Config defaults ────────────────────────────────────────────────────────────

SLM_BASE_URL = "http://codon-gpu-007.ebi.ac.uk:8000/v1/"
SLM_MODEL = "qwen3.5-9b"
LLM_MODEL = "claude-haiku-4-5-20251001"
MLP_MODEL_DIR_DEFAULT = "results/ablation/models/tau_0.50_seed_42"

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_env(project_root: Path) -> dict:
    env_path = project_root / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f".env not found at {env_path}")
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def parse_score(response_text: str) -> float | None:
    """Extract score from model response JSON block.

    Handles both legacy key "Sufficient Support Score" and the prompt
    template's actual key "score" or "sufficiency_score".
    """
    for pattern in (
        r'"Sufficient Support Score"\s*:\s*([0-9]*\.?[0-9]+)',
        r'"score"\s*:\s*([0-9]*\.?[0-9]+)',
        r'"sufficiency_score"\s*:\s*([0-9]*\.?[0-9]+)',
    ):
        matches = re.findall(pattern, response_text)
        if matches:
            return float(matches[-1])
    return None


# ── SLM / LLM callers ─────────────────────────────────────────────────────────

def call_slm(prompt: str, base_url: str, model: str, api_key: str = "EMPTY") -> str:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096,
        stream=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (response.choices[0].message.content or "").strip()


def call_llm(prompt: str, model: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


# ── MLP backend ────────────────────────────────────────────────────────────────

def load_mlp(model_dir: Path):
    """Load the MLP model, scaler params, and expected features from model_dir."""
    import numpy as np
    import torch
    import torch.nn as nn

    with open(model_dir / "mlp_config.json") as f:
        config = json.load(f)

    # Re-define SufficiencyMLP inline (same architecture as test_mlp_classifier.py)
    class SufficiencyMLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            return self.network(x)

    model = SufficiencyMLP(input_dim=config["input_dim"], hidden_dim=config.get("hidden_dim", 64))
    weights_path = model_dir / "mlp_classifier_weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    return {
        "model": model,
        "expected_features": config["expected_features"],
        "mean": np.array(config["scaler_mean"]),
        "scale": np.array(config["scaler_scale"]),
        "config": config,
    }


def evidence_state_to_papers_dicts(data: dict) -> tuple[list[dict], int]:
    """
    Convert evidence_state.json data into the format expected by FeatureAggregator.

    Returns:
        (papers_dicts, num_full_text)
        where papers_dicts is [{metadata_features: {...}, nlp_features: {...}}, ...]
    """
    papers = data.get("papers") or {}
    papers_dicts = []
    num_full_text = 0

    for pmid, paper in papers.items():
        meta = paper.get("metadata") or {}
        nlp = paper.get("nlp") or {}

        metadata_features = {
            "publication_year": meta.get("publication_year"),
            "log_impact_factor": meta.get("log_impact_factor"),
            "normalized_citation_count": meta.get("normalized_citation_count"),
            "author_h_index_max": meta.get("author_h_index_max"),
        }
        # Only include if at least one metadata field is populated
        metadata_features = metadata_features if any(v is not None for v in metadata_features.values()) else None

        nlp_features = {
            "nli_entailment": nlp.get("nli_entailment"),
            "nli_contradiction": nlp.get("nli_contradiction"),
            "nli_neutral": nlp.get("nli_neutral"),
            "claim_entity_coverage": nlp.get("claim_entity_coverage"),
            "semantic_similarity": nlp.get("semantic_similarity"),
        }
        nlp_features = nlp_features if any(v is not None for v in nlp_features.values()) else None

        papers_dicts.append({
            "metadata_features": metadata_features,
            "nlp_features": nlp_features,
        })

        if paper.get("full_text") is not None:
            num_full_text += 1

    return papers_dicts, num_full_text


def call_mlp(evidence_state_path: Path, mlp: dict, aggregator) -> dict:
    """Run MLP inference on a single evidence_state.json.

    Returns a result dict with sufficient_support_score (probability [0,1]).
    """
    import numpy as np
    import torch

    with open(evidence_state_path, encoding="utf-8") as f:
        data = json.load(f)

    papers_dicts, num_full_text = evidence_state_to_papers_dicts(data)

    # Aggregate features across papers
    nested = aggregator.aggregate_all(papers_dicts)

    # Flatten nested dict
    _NESTED_KEYS = ("metadata_aggregation", "nlp_aggregation", "cross_features")
    flat: dict = {}
    for sub_key in _NESTED_KEYS:
        if sub_key in nested:
            flat.update(nested[sub_key])

    flat["num_full_text"] = num_full_text

    # Vectorize in expected feature order, imputing 0.0 for missing
    vec = []
    for fname in mlp["expected_features"]:
        val = flat.get(fname)
        vec.append(float(val) if isinstance(val, (int, float)) else 0.0)

    X_raw = np.array([vec])
    X_scaled = (X_raw - mlp["mean"]) / mlp["scale"]
    X_tensor = torch.FloatTensor(X_scaled)

    with torch.no_grad():
        logit = mlp["model"](X_tensor)
        prob = torch.sigmoid(logit).item()

    return {
        "sufficient_support_score": round(prob, 6),
        "logit": round(logit.item(), 6),
        "flat_features": {k: flat.get(k) for k in mlp["expected_features"]},
        "num_papers": flat.get("num_papers", len(papers_dicts)),
        "num_full_text": num_full_text,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run SLM/LLM/MLP inference for the sufficiency ablation.")
    parser.add_argument("--backend", choices=["slm", "llm", "mlp"], required=True,
                        help="slm=Qwen/vLLM  llm=Claude Haiku  mlp=existing MLP classifier")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Source RLM run dir containing evidence_state.json files. "
                             "Used by MLP backend. Default: results/signor_eval_20260330_205939")
    parser.add_argument("--ablation-dir", type=str, default=None,
                        help="Ablation output dir. "
                             "Default: results/slm_llm_ablation/signor_eval_20260330_205939")
    parser.add_argument("--mlp-model-dir", type=str, default=None,
                        help=f"MLP model directory. Default: {MLP_MODEL_DIR_DEFAULT}")
    parser.add_argument("--slm-base-url", type=str, default=SLM_BASE_URL)
    parser.add_argument("--slm-model", type=str, default=SLM_MODEL)
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first item and exit without calling any model.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing result files instead of skipping them.")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between API calls for slm/llm (default: 1.0).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent

    base_dir = Path(args.base_dir).resolve() if args.base_dir else \
               project_root / "results" / "signor_eval_20260330_205939"
    ablation_dir = Path(args.ablation_dir).resolve() if args.ablation_dir else \
                   project_root / "results" / "slm_llm_ablation" / "signor_eval_20260330_205939"
    mlp_model_dir = Path(args.mlp_model_dir).resolve() if args.mlp_model_dir else \
                    project_root / MLP_MODEL_DIR_DEFAULT

    # ── MLP backend ──────────────────────────────────────────────────────────
    if args.backend == "mlp":
        # Add scripts dir to path so FeatureAggregator is importable
        scripts_dir = project_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from sufficiency_classifier.feature_aggregation import FeatureAggregator

        if not mlp_model_dir.exists():
            print(f"Error: MLP model dir not found: {mlp_model_dir}")
            return

        print(f"Loading MLP from: {mlp_model_dir}")
        mlp = load_mlp(mlp_model_dir)
        aggregator = FeatureAggregator()
        print(f"Expected features ({len(mlp['expected_features'])}): {mlp['expected_features']}")

        evidence_files = sorted(glob.glob(
            str(base_dir / "**" / "evidence_state.json"), recursive=True
        ))
        print(f"Found {len(evidence_files)} evidence_state.json files in {base_dir}")

        if args.dry_run:
            ep = Path(evidence_files[0])
            print(f"\n--- DRY RUN: {ep.relative_to(base_dir)} ---")
            result = call_mlp(ep, mlp, aggregator)
            print(json.dumps(result, indent=2))
            return

        total = len(evidence_files)
        success = 0
        for i, ev_path in enumerate(evidence_files):
            ev_path = Path(ev_path)
            rel = ev_path.parent.relative_to(base_dir)
            dest_dir = ablation_dir / rel
            dest_dir.mkdir(parents=True, exist_ok=True)
            result_path = dest_dir / "mlp_sufficiency_result.json"

            if result_path.exists() and not args.overwrite:
                print(f"[{i+1}/{total}] Skip (exists): {rel}")
                success += 1
                continue

            print(f"[{i+1}/{total}] MLP: {rel} ...", end=" ", flush=True)
            start = time.time()
            try:
                result = call_mlp(ev_path, mlp, aggregator)
                elapsed = time.time() - start
                result["backend"] = "mlp"
                result["model"] = str(mlp_model_dir.name)
                result["duration_seconds"] = round(elapsed, 2)
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"score={result['sufficient_support_score']:.4f} ({elapsed:.2f}s)")
                success += 1
            except Exception as e:
                elapsed = time.time() - start
                print(f"ERROR ({elapsed:.2f}s): {e}")
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump({"backend": "mlp", "error": str(e), "sufficient_support_score": None}, f, indent=2)

        print(f"\nDone. {success}/{total} completed.")
        return

    # ── SLM / LLM backend ────────────────────────────────────────────────────
    if not ablation_dir.exists():
        print(f"Error: ablation dir not found: {ablation_dir}")
        print("Run generate_sufficiency_prompts.py first.")
        return

    env = load_env(project_root)
    api_key = env.get("CLAUDE_API_KEY") or env.get("ANTHROPIC_API_KEY") or "EMPTY"

    prompt_files = sorted(glob.glob(
        str(ablation_dir / "**" / "llm_sufficiency_prompt.txt"), recursive=True
    ))
    print(f"Found {len(prompt_files)} prompt files in {ablation_dir}")

    if args.dry_run:
        print(f"\n--- DRY RUN: {prompt_files[0]} ---")
        print(open(prompt_files[0]).read()[:1000])
        print("...[truncated]")
        return

    result_filename = f"{args.backend}_sufficiency_result.json"
    total = len(prompt_files)
    success = 0
    parse_failures = 0

    for i, prompt_path in enumerate(prompt_files):
        prompt_path = Path(prompt_path)
        result_path = prompt_path.parent / result_filename
        rel = prompt_path.parent.relative_to(ablation_dir)

        if result_path.exists() and not args.overwrite:
            print(f"[{i+1}/{total}] Skip (exists): {rel}")
            success += 1
            continue

        prompt_text = open(prompt_path, encoding="utf-8").read()
        print(f"[{i+1}/{total}] {args.backend.upper()}: {rel} ...", end=" ", flush=True)

        start = time.time()
        try:
            if args.backend == "slm":
                response = call_slm(prompt_text, args.slm_base_url, args.slm_model, api_key)
                model_name = args.slm_model
            else:
                response = call_llm(prompt_text, args.llm_model, api_key)
                model_name = args.llm_model

            elapsed = time.time() - start
            score = parse_score(response)

            result = {
                "backend": args.backend,
                "model": model_name,
                "sufficient_support_score": score,
                "response": response,
                "parse_success": score is not None,
                "duration_seconds": round(elapsed, 2),
            }
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            status = f"score={score}" if score is not None else "PARSE_FAIL"
            print(f"{status} ({elapsed:.1f}s)")
            success += 1
            if score is None:
                parse_failures += 1

        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR ({elapsed:.1f}s): {e}")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({
                    "backend": args.backend, "model": args.slm_model if args.backend == "slm" else args.llm_model,
                    "sufficient_support_score": None, "response": None,
                    "parse_success": False, "error": str(e), "duration_seconds": round(elapsed, 2),
                }, f, indent=2)

        if i < total - 1:
            time.sleep(args.delay)

    print(f"\nDone. {success}/{total} completed, {parse_failures} score parse failures.")
    print(f"Results written as '{result_filename}' alongside each prompt.")


if __name__ == "__main__":
    main()
