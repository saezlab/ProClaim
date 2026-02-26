"""
Script to test the trained MLP classifier on a specific claim_id from the training data.

Usage:
# important features model
uv run scripts/sufficiency_classifier/test_mlp_classifier.py --claim_id 7 --model_dir results/models/classifier_important

# minimal features model
uv run scripts/sufficiency_classifier/test_mlp_classifier.py --claim_id 7 --model_dir results/models/classifier_minimal

"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(message)s')

class SufficiencyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        # Note: No final Sigmoid — model outputs raw logits (matches training with BCEWithLogitsLoss)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim_id", type=int, required=True, help="Claim ID to test")
    parser.add_argument("--data_path", type=str, default="data/classifier_train_data.json", help="Path to training data JSON")
    parser.add_argument("--model_dir", type=str, default="results/models/classifier", help="Directory containing the model and config")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.model_dir) / "mlp_config.json"
    if not config_path.exists():
        logging.error(f"Config not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        logging.error(f"Data not found at {data_path}")
        return

    with open(data_path, "r") as f:
        data = json.load(f)

    # Find ALL entries for the given claim_id
    matching_items = [item for item in data if item.get("claim_id") == args.claim_id]

    if not matching_items:
        logging.error(f"Claim ID {args.claim_id} not found in {args.data_path}")
        return

    # Print claim info (claim_text is the same across entries)
    claim_text = matching_items[0].get("claim_text", "N/A")
    print(f"=== Claim ID: {args.claim_id} ({len(matching_items)} entries) ===")
    print(f"Claim: {claim_text}\n")

    # Load Model once
    expected_features = config["expected_features"]
    input_dim = config["input_dim"]
    hidden_dim = config.get("hidden_dim", 64)
    mean = np.array(config["scaler_mean"])
    scale = np.array(config["scaler_scale"])

    model = SufficiencyMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    model_path = Path(args.model_dir) / "mlp_classifier_weights.pth"

    if not model_path.exists():
        logging.error(f"Model weights not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Predict on each entry
    print(f"{'#':<4} {'pool_type':<22} {'true_y':<8} {'pred':<8} {'prob_1':<10} {'logit':<10}")
    print("-" * 62)

    for idx, item in enumerate(matching_items):
        feats = item.get("features", {})

        # Support both flat and nested feature formats
        flat_feats = {}
        has_nested = any(k in feats for k in ("metadata_aggregation", "nlp_aggregation", "cross_features"))
        if has_nested:
            for sub_key in ("metadata_aggregation", "nlp_aggregation", "cross_features"):
                if sub_key in feats:
                    flat_feats.update(feats[sub_key])
        else:
            flat_feats = feats

        vec = []
        for fname in expected_features:
            val = flat_feats.get(fname)
            if val is None or not isinstance(val, (int, float)):
                vec.append(0.0)
            else:
                vec.append(float(val))

        X_raw = np.array([vec])
        X_scaled = (X_raw - mean) / scale
        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            logit = model(X_tensor)
            prob_1 = torch.sigmoid(logit).item()

        pred = 1 if prob_1 > 0.5 else 0
        true_y = item.get("target_y")
        pool_type = item.get("pool_type", "unknown")
        correct = "✓" if pred == true_y else "✗"

        print(f"{idx:<4} {pool_type:<22} {true_y:<8} {pred:<8} {prob_1:<10.4f} {logit.item():<10.4f} {correct}")
    
if __name__ == "__main__":
    main()
