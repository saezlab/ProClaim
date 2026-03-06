"""
Model selection script: compares Logistic Regression, small MLP, and large MLP
across different feature presets using the same train/val/test split.

Usage:
  uv run scripts/sufficiency_classifier/model_selection.py --data_path data/classifier_train_data.json

  uv run scripts/sufficiency_classifier/model_selection.py --data_path data/classifier_train_data.json > results/models/classifier_best/model_selection.log 2>&1
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# Feature presets (copied from train_mlp_classifier.py for self-containedness)
# ---------------------------------------------------------------------------

ALL_FEATURES = [
    "num_papers", "num_papers_with_metadata", "num_full_text",
    "max_log_IF", "mean_log_IF",
    "max_h_index", "avg_max_h_index", "max_norm_citation", "mean_norm_citation",
    "latest_year_age", "year_span",
    "entailment_ratio", "contradiction_ratio", "controversy_index",
    "max_entity_coverage", "mean_entity_coverage", "max_similarity", "mean_similarity",
    "weighted_entailment_IF", "weighted_contradiction_IF",
    "weighted_entailment_citation", "weighted_contradiction_citation",
    "weighted_entailment_temporal", "weighted_contradiction_temporal"
]

IMPORTANT_FEATURES = [
    "mean_similarity", "num_full_text", "year_span",
    "mean_entity_coverage", "controversy_index",
    "weighted_entailment_temporal", "mean_norm_citation",
    "mean_log_IF", "contradiction_ratio", "num_papers",
]

MINIMAL_FEATURES = [
    "mean_similarity", "num_full_text", "year_span",
    "mean_entity_coverage", "controversy_index",
]

FEATURE_PRESETS = {
    "all": ALL_FEATURES,
    "important": IMPORTANT_FEATURES,
    "minimal": MINIMAL_FEATURES,
}

# ---------------------------------------------------------------------------
# Data loading (same logic as train_mlp_classifier.py)
# ---------------------------------------------------------------------------

class SufficiencyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_raw_data(filepath: str, feature_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load JSON data and return unscaled X, y."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    X_list, y_list = [], []
    for item in data:
        target = item["target_y"]
        feats = item.get("features", {})

        flat_feats = {}
        has_nested = any(k in feats for k in ("metadata_aggregation", "nlp_aggregation", "cross_features"))
        if has_nested:
            for sub_key in ("metadata_aggregation", "nlp_aggregation", "cross_features"):
                if sub_key in feats:
                    flat_feats.update(feats[sub_key])
        else:
            flat_feats = feats

        vec = []
        for fname in feature_list:
            val = flat_feats.get(fname)
            if val is None or not isinstance(val, (int, float)):
                vec.append(0.0)
            else:
                vec.append(float(val))

        X_list.append(vec)
        y_list.append(target)

    return np.array(X_list), np.array(y_list)


# ---------------------------------------------------------------------------
# PyTorch model definitions
# ---------------------------------------------------------------------------

class SmallMLP(nn.Module):
    """Single hidden layer, no BatchNorm, no Dropout."""
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x)


class LargeMLP(nn.Module):
    """Two hidden layers with BatchNorm + Dropout (same as existing)."""
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


# ---------------------------------------------------------------------------
# Training helper for PyTorch models
# ---------------------------------------------------------------------------

def train_pytorch_model(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 100, lr: float = 1e-3, batch_size: int = 32,
    patience: int = 15, pos_weight: float = 1.0,
) -> nn.Module:
    """Train a PyTorch model and return the best checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_ds = SufficiencyDataset(X_train, y_train)
    val_ds = SufficiencyDataset(X_val, y_val)

    bs = min(batch_size, len(X_train))
    drop_last = len(X_train) > bs
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    pw = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                loss = criterion(model(bx), by)
                val_loss += loss.item() * bx.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_pytorch(model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate a PyTorch model on a dataset, return metrics dict."""
    device = next(model.parameters()).device
    ds = SufficiencyDataset(X, y)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    preds, targets = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            out = model(bx)
            preds.extend((out > 0.0).float().cpu().numpy().flatten())
            targets.extend(by.numpy().flatten())

    acc = accuracy_score(targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Model selection: compare LR, small MLP, large MLP across feature presets")
    parser.add_argument("--data_path", type=str, default="data/classifier_train_data.json")
    parser.add_argument("--output_dir", type=str, default="results/models/classifier_best")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data_file = Path(args.data_path)
    if not data_file.exists():
        logging.error(f"Data file not found: {data_file}")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_types = ["logistic", "mlp_small", "mlp_large"]
    feature_presets = ["all", "important", "minimal"]

    all_results: List[Dict[str, Any]] = []
    best_f1 = -1.0
    best_info: Dict[str, Any] = {}

    for feat_name in feature_presets:
        feat_list = FEATURE_PRESETS[feat_name]
        logging.info(f"\n{'='*60}")
        logging.info(f"Feature preset: {feat_name} ({len(feat_list)} features)")
        logging.info(f"{'='*60}")

        # Load data (unscaled)
        X_raw, y = load_raw_data(str(data_file), feat_list)
        logging.info(f"Dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")
        logging.info(f"Labels: y=1 {int(y.sum())}, y=0 {int(len(y)-y.sum())}")

        # Same split for all models
        X_tv, X_test, y_tv, y_test = train_test_split(X_raw, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.1765, random_state=42, stratify=y_tv)

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        # Class weight
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = n_neg / max(n_pos, 1)

        for model_name in model_types:
            logging.info(f"\n--- {model_name} | {feat_name} ---")

            if model_name == "logistic":
                clf = LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42,
                    solver='lbfgs',
                )
                clf.fit(X_train_s, y_train)
                test_preds = clf.predict(X_test_s)

                acc = accuracy_score(y_test, test_preds)
                prec, rec, f1, _ = precision_recall_fscore_support(y_test, test_preds, average='binary', zero_division=0)
                metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

            elif model_name == "mlp_small":
                model = SmallMLP(input_dim=len(feat_list), hidden_dim=16)
                model = train_pytorch_model(
                    model, X_train_s, y_train, X_val_s, y_val,
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                    patience=15, pos_weight=pos_weight,
                )
                metrics = evaluate_pytorch(model, X_test_s, y_test)

            elif model_name == "mlp_large":
                model = LargeMLP(input_dim=len(feat_list), hidden_dim=64)
                model = train_pytorch_model(
                    model, X_train_s, y_train, X_val_s, y_val,
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                    patience=15, pos_weight=pos_weight,
                )
                metrics = evaluate_pytorch(model, X_test_s, y_test)

            result = {
                "model": model_name,
                "features": feat_name,
                "n_features": len(feat_list),
                **metrics,
            }
            all_results.append(result)

            logging.info(f"  Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
                         f"Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

            # Track best
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_info = {
                    "result": result,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "feature_list": feat_list,
                }
                # Save best model artifacts
                if model_name == "logistic":
                    import pickle
                    with open(out_dir / "best_model.pkl", "wb") as f:
                        pickle.dump(clf, f)
                    best_info["model_file"] = "best_model.pkl"
                    best_info["model_class"] = "LogisticRegression"
                else:
                    torch.save(model.state_dict(), out_dir / "best_model.pth")
                    best_info["model_file"] = "best_model.pth"
                    best_info["model_class"] = model_name
                    best_info["hidden_dim"] = 16 if model_name == "mlp_small" else 64

    # ---------------------------------------------------------------------------
    # Print comparison table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODEL SELECTION RESULTS")
    print("=" * 80)
    header = f"{'Model':<14} {'Features':<12} {'#Feat':>5}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}"
    print(header)
    print("-" * 80)

    for r in all_results:
        marker = " *" if r["f1"] == best_f1 else ""
        print(f"{r['model']:<14} {r['features']:<12} {r['n_features']:>5}  "
              f"{r['accuracy']:>7.4f}  {r['precision']:>7.4f}  "
              f"{r['recall']:>7.4f}  {r['f1']:>7.4f}{marker}")

    print("-" * 80)
    print(f"* Best model: {best_info['result']['model']} ({best_info['result']['features']}) — F1={best_f1:.4f}")
    print("=" * 80)

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    summary = {
        "all_results": all_results,
        "best": best_info["result"],
    }
    summary_path = out_dir / "model_selection_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save best model config (compatible with test_mlp_classifier.py)
    config = {
        "model_type": best_info["result"]["model"],
        "input_dim": best_info["result"]["n_features"],
        "expected_features": best_info["feature_list"],
        "scaler_mean": best_info["scaler_mean"],
        "scaler_scale": best_info["scaler_scale"],
        "metrics": {
            "test_accuracy": best_info["result"]["accuracy"],
            "test_f1": best_info["result"]["f1"],
            "test_precision": best_info["result"]["precision"],
            "test_recall": best_info["result"]["recall"],
        }
    }
    if "hidden_dim" in best_info:
        config["hidden_dim"] = best_info["hidden_dim"]

    config_path = out_dir / "mlp_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logging.info(f"\nSaved results to {summary_path}")
    logging.info(f"Saved best model config to {config_path}")
    logging.info(f"Saved best model weights to {out_dir / best_info['model_file']}")


if __name__ == "__main__":
    main()
