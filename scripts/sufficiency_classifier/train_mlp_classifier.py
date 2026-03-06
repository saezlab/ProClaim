"""
Script to train an MLP classifier for predicting the Sufficiency Score
of an aggregated evidence pool, as defined in classifier_plan.md.

uv run scripts/sufficiency_classifier/train_mlp_classifier.py \
  --data_path data/classifier_train_data.json \
  --output_dir results/models/classifier \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-3

uv run scripts/sufficiency_classifier/train_mlp_classifier.py --data_path data/classifier_train_data.json > results/models/classifier/train.log 2>&1


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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default features to expect from aggregation
ALL_FEATURES = [
    # Metadata
    "num_papers", "num_papers_with_metadata", "num_full_text",
    "max_log_IF", "mean_log_IF",
    "max_h_index", "avg_max_h_index", "max_norm_citation", "mean_norm_citation",
    "latest_year_age", "year_span",
    # NLP
    "entailment_ratio", "contradiction_ratio", "controversy_index",
    "max_entity_coverage", "mean_entity_coverage", "max_similarity", "mean_similarity",
    # Cross-Features
    "weighted_entailment_IF", "weighted_contradiction_IF",
    "weighted_entailment_citation", "weighted_contradiction_citation",
    "weighted_entailment_temporal", "weighted_contradiction_temporal"
]

# Top features by Cohen's d, removing one from each highly correlated pair (r>0.9)
# Kept: mean_similarity (not max), num_full_text (not num_papers_with_metadata),
#       mean_entity_coverage (not max), mean_log_IF (not max), mean_norm_citation (not max),
#       avg_max_h_index (not max)
IMPORTANT_FEATURES = [
    "mean_similarity",          # d=1.27  (strongest)
    "num_full_text",            # d=0.84
    "year_span",                # d=0.72
    "mean_entity_coverage",     # d=0.70
    "controversy_index",        # d=0.43
    "weighted_entailment_temporal",  # d=0.36
    "mean_norm_citation",       # d=0.36
    "mean_log_IF",              # d=0.31
    "contradiction_ratio",      # d=0.21
    "num_papers",               # d=0.22
]

MINIMAL_FEATURES = [
    "mean_similarity",          # d=1.27
    "num_full_text",            # d=0.84
    "year_span",                # d=0.72
    "mean_entity_coverage",     # d=0.70
    "controversy_index",        # d=0.43
]

FEATURE_PRESETS = {
    "all": ALL_FEATURES,
    "important": IMPORTANT_FEATURES,
    "minimal": MINIMAL_FEATURES,
}

class SufficiencyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1) # shape (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SufficiencyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        # Note: No final Sigmoid — use BCEWithLogitsLoss for numerical stability
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

def load_and_preprocess_data(filepath: str, feature_list: List[str] = None) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Loads the JSON data and flattens it into X (features) and y (targets)."""
    if feature_list is None:
        feature_list = ALL_FEATURES

    logging.info(f"Loading training data from {filepath}")
    logging.info(f"Using {len(feature_list)} features: {feature_list}")
    with open(filepath, 'r') as f:
        data = json.load(f)

    X_list = []
    y_list = []

    for item in data:
        target = item["target_y"]
        feats = item.get("features", {})
        
        # Support both flat and nested feature formats
        flat_feats = {}
        has_nested = any(k in feats for k in ("metadata_aggregation", "nlp_aggregation", "cross_features"))
        if has_nested:
            for sub_key in ("metadata_aggregation", "nlp_aggregation", "cross_features"):
                if sub_key in feats:
                    flat_feats.update(feats[sub_key])
        else:
            flat_feats = feats  # Features are already flat

        # Create numerical vector aligned with EXPECTED_FEATURES
        vec = []
        for fname in feature_list:
            val = flat_feats.get(fname)
            # Impute missing with 0 (or could use mean imputation)
            if val is None or not isinstance(val, (int, float)):
                vec.append(0.0)
            else:
                vec.append(float(val))
                
        X_list.append(vec)
        y_list.append(target)

    X = np.array(X_list)
    y = np.array(y_list)

    logging.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logging.info(f"Label distribution: y=1: {int(y.sum())}/{len(y)} ({y.mean()*100:.1f}%), y=0: {int(len(y)-y.sum())}/{len(y)} ({(1-y.mean())*100:.1f}%)")

    # Diagnostic: log non-zero feature counts to verify features loaded correctly
    non_zero_counts = (X != 0).sum(axis=0)
    for i, fname in enumerate(feature_list):
        logging.info(f"  Feature '{fname}': {non_zero_counts[i]}/{len(X)} non-zero")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    epochs: int, 
    lr: float,
    device: torch.device,
    patience: int = 15,
    pos_weight: float = 1.0,
) -> nn.Module:
    
    # BCEWithLogitsLoss for numerical stability (model outputs raw logits)
    pw = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                
                preds = (outputs > 0.0).float()  # logits: >0 means positive class
                val_preds.extend(preds.cpu().numpy().flatten())
                val_targets.extend(batch_y.cpu().numpy().flatten())

        val_loss = val_loss / len(val_loader.dataset)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Calculate metrics
        acc = accuracy_score(val_targets, val_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='binary', zero_division=0)

        # Save best model and Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            logging.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {acc:.4f} - F1: {f1:.4f} - LR: {current_lr:.6f} (New Best)")
        else:
            epochs_no_improve += 1
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {acc:.4f} - F1: {f1:.4f} - LR: {current_lr:.6f}")
            
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1} (No improvement for {patience} epochs)")
                break

    # Load best weights
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/classifier_train_data.json", help="Path to training data JSON")
    parser.add_argument("--output_dir", type=str, default="results/models/classifier", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--features", type=str, default="all",
                        help="Feature preset ('all', 'important', 'minimal') or comma-separated feature names")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logging.info(f"Random seed set to: {args.seed}")

    # Resolve feature list
    if args.features in FEATURE_PRESETS:
        feature_list = FEATURE_PRESETS[args.features]
        logging.info(f"Using feature preset: '{args.features}' ({len(feature_list)} features)")
    else:
        feature_list = [f.strip() for f in args.features.split(',')]
        logging.info(f"Using custom features: {feature_list}")

    data_file = Path(args.data_path)
    if not data_file.exists():
        logging.error(f"Data file not found: {data_file}")
        logging.info("Please generate data first using scripts/sufficiency_classifier/generate_classifier_data.py")
        return

    # 1. Load Data
    X, y, scaler = load_and_preprocess_data(str(data_file), feature_list=feature_list)

    if len(X) < 10:
        logging.error(f"Dataset too small ({len(X)} samples). Need more data to train.")
        return

    # Split: Train(70%), Val(15%), Test(15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=args.seed, stratify=y)
    # remaining 85%, doing 15/85 ~ 0.176 to get 15% overall for val
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=args.seed, stratify=y_train_val)
    
    logging.info(f"Split sizes -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    logging.info(f"Positives -> Train: {sum(y_train)} | Val: {sum(y_val)} | Test: {sum(y_test)}")

    # Datasets
    train_dataset = SufficiencyDataset(X_train, y_train)
    val_dataset = SufficiencyDataset(X_val, y_val)
    test_dataset = SufficiencyDataset(X_test, y_test)

    # Loaders
    # Handle small datasets gracefully during testing
    bs = min(args.batch_size, len(X_train))
    # Drop last only if dataset is larger than batch size, avoids error if dataset is exactly 1 batch or less
    drop_last = len(X_train) > bs
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    input_dim = X.shape[1]
    model = SufficiencyMLP(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)

    # 3. Train
    # Calculate pos_weight for class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    logging.info(f"Class balance -> pos_weight: {pos_weight:.3f} (neg/pos = {n_neg}/{int(n_pos)})")

    logging.info("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        pos_weight=pos_weight
    )

    # 4. Final Evaluation (On Test Set)
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = (outputs > 0.0).float()  # logits: >0 means positive class
            test_preds.extend(preds.cpu().numpy().flatten())
            test_targets.extend(batch_y.numpy().flatten())

    acc = accuracy_score(test_targets, test_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='binary', zero_division=0)
    
    logging.info("\n--- Final Test Set Metrics ---")
    logging.info(f"Accuracy:  {acc:.4f}")
    logging.info(f"Precision: {prec:.4f}")
    logging.info(f"Recall:    {rec:.4f}")
    logging.info(f"F1 Score:  {f1:.4f}")

    # 5. Save model and metadata
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch weights
    model_path = out_dir / "mlp_classifier_weights.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save Feature configuration (so inference script knows the exact order)
    config = {
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "expected_features": feature_list,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "metrics": {
            "test_accuracy": float(acc),
            "test_f1": float(f1),
            "test_precision": float(prec),
            "test_recall": float(rec)
        }
    }
    
    config_path = out_dir / "mlp_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        
    logging.info(f"Saved model weights to {model_path}")
    logging.info(f"Saved model config to {config_path}")

if __name__ == "__main__":
    main()
