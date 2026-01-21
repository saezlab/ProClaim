import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# --- Model Definition and Training ---
class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = GATv2Conv(hidden_channels, out_channels)

        self.decoder_linear = nn.Linear(2 * out_channels, out_channels)
        self.out_linear = nn.Linear(out_channels, 1)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        out = self.out_linear(self.decoder_linear(torch.cat([src, dst], dim=1)).relu()).squeeze()
        return out

def get_f1_score(labels, preds):
    # Get precision and recall for various thresholds
    precision, recall, thresholds = precision_recall_curve(labels, preds)

    # Calculate F1 score for each threshold, avoiding division by zero
    # We slice the arrays to match the length of the thresholds array
    p, r = precision[:-1], recall[:-1]
    f1_scores = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) != 0)

    # Find the threshold that gives the maximum F1 score
    best_f1_idx = np.argmax(f1_scores)
    optimal_f1 = f1_scores[best_f1_idx]
    return optimal_f1

def edges_to_tensor(edge_set, node_to_idx):
    """Convert a set of (source, target) tuples to edge_index tensor."""
    edge_list = []
    for src, tgt in edge_set:
        if src in node_to_idx and tgt in node_to_idx:
            edge_list.append([node_to_idx[src], node_to_idx[tgt]])

    if len(edge_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def noise_positive_edges(positive_edge_index):
    """Generate negative edges by randomly sampling node pairs that don't exist in positive_edge_index."""
    existing_edges = set(tuple(e) for e in positive_edge_index.t().tolist())
    existing_nodes = sorted(list(set(positive_edge_index.flatten().tolist())))

    num_negative_edges = positive_edge_index.shape[1]
    negative_edges_list = []
    negative_edges_set = set()

    while len(negative_edges_list) < num_negative_edges:
        u, v = np.random.randint(0, len(existing_nodes), 2)
        u, v = existing_nodes[u], existing_nodes[v]
        if u != v and (u, v) not in existing_edges and (u, v) not in negative_edges_set:
            negative_edges_list.append([u, v])
            negative_edges_set.add((u, v))

    negative_edge_index = torch.tensor(negative_edges_list, dtype=torch.long).t().contiguous()
    return negative_edge_index


def main(args):
    # --- Load and Preprocess Data ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 1. Load Full Graph and Test Data ===
    print("\n--- Loading Graphs ---")
    
    # Get absolute paths to data files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../../'))
    data_dir = os.path.join(project_root, 'data', 'signor')
    
    full_graph_path = os.path.join(data_dir, 'Oct2025_release.txt')
    pos_edges_path = os.path.join(data_dir, 'true_positive_edges.csv')
    neg_edges_path = os.path.join(data_dir, 'true_negative_edges.csv')

    print(f"Loading data from: {data_dir}")

    full_df = pd.read_csv(full_graph_path, sep='\t')
    full_df = full_df[(full_df['TYPEA'] == 'protein') & (full_df['TYPEB'] == 'protein')].copy()
    full_df = full_df.dropna(subset=['ENTITYA', 'ENTITYB'])
    full_df = full_df[full_df['ENTITYA'] != full_df['ENTITYB']]  # Remove self-loops
    print(f"Original full graph: {len(full_df)} interactions")

    # Load Test Data Early to Filter
    print("Loading True Labels for Test Set (to exclude from training)...")
    pos_df = pd.read_csv(pos_edges_path)
    neg_df = pd.read_csv(neg_edges_path)

    # Identify test edges to remove
    test_edges_set = set(zip(pos_df['ENTITYA'], pos_df['ENTITYB']))

    # Filter full_df to remove test edges
    initial_len = len(full_df)
    full_df['is_test'] = [ (a, b) in test_edges_set for a, b in zip(full_df['ENTITYA'], full_df['ENTITYB']) ]
    full_df = full_df[~full_df['is_test']].drop(columns=['is_test'])
    print(f"Filtered graph: {len(full_df)} interactions (removed {initial_len - len(full_df)} test edges)")

    # === 2. Create Node Mapping ===
    # Use union of all nodes to ensure consistency
    all_nodes = pd.unique(full_df[['ENTITYA', 'ENTITYB']].values.ravel())
    test_nodes = pd.unique(pd.concat([pos_df, neg_df])[['ENTITYA', 'ENTITYB']].values.ravel())
    
    # Ensure all are strings and remove NaNs
    all_nodes = np.concatenate([all_nodes, test_nodes])
    all_nodes = [str(x) for x in all_nodes if pd.notna(x)]
    all_nodes = np.unique(all_nodes)

    node_to_idx = {name: i for i, name in enumerate(all_nodes)}
    num_nodes = len(all_nodes)
    print(f"Total nodes: {num_nodes}")

    # === 3. Convert Filtered Graph to Edge Index ===
    source_nodes = [node_to_idx[name] for name in full_df['ENTITYA']]
    target_nodes = [node_to_idx[name] for name in full_df['ENTITYB']]
    positive_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    print(f"Positive edges (training pool): {positive_edge_index.shape[1]}")

    # === 4. Generate Random Negative Edges for Training ===
    print("\n--- Generating Random Negative Edges for Train/Val ---")
    negative_edge_index = noise_positive_edges(positive_edge_index)
    print(f"Negative edges (random): {negative_edge_index.shape[1]}")

    # === 5. Combine Positive and Negative for Train/Val ===
    all_train_val_edges = torch.cat([positive_edge_index, negative_edge_index], dim=1)
    all_train_val_labels = torch.cat([
        torch.ones(positive_edge_index.shape[1]),
        torch.zeros(negative_edge_index.shape[1])
    ])
    print(f"Total train/val edges: {all_train_val_edges.shape[1]}")
    print(f"Train/val balance: {all_train_val_labels.sum().item():.0f} positives / {(1-all_train_val_labels).sum().item():.0f} negatives")

    # === 6. Load True Positive/Negative Edges (Already loaded) ===
    print("\n--- Preparing Test Set ---")
    # pos_df and neg_df are already loaded
    print(f"True positives: {len(pos_df)} edges")
    print(f"True negatives: {len(neg_df)} edges")

    # Convert to edge indices
    true_pos_edges = set(zip(pos_df['ENTITYA'], pos_df['ENTITYB']))
    true_neg_edges = set(zip(neg_df['ENTITYA'], neg_df['ENTITYB']))

    test_pos_edge_index = edges_to_tensor(true_pos_edges, node_to_idx)
    test_neg_edge_index = edges_to_tensor(true_neg_edges, node_to_idx)

    test_edge_index = torch.cat([test_pos_edge_index, test_neg_edge_index], dim=1)
    test_labels = torch.cat([
        torch.ones(test_pos_edge_index.shape[1]),
        torch.zeros(test_neg_edge_index.shape[1])
    ])
    print(f"Test set: {test_edge_index.shape[1]} edges ({test_labels.sum().item():.0f} pos / {(1-test_labels).sum().item():.0f} neg)")

    # === 7. Feature Generation ===
    if args.use_text_embeddings:
        print(f"\n--- Loading Text Embeddings: {args.embedding_path} ---")
        node_features = [None for _ in range(num_nodes)]
        gene_dict = np.load(args.embedding_path, allow_pickle=True).item()
        for gene, embedding in gene_dict.items():
            if gene in node_to_idx:
                node_features[node_to_idx[gene]] = embedding

        # Check if any nodes are missing embeddings
        missing_count = sum(1 for x in node_features if x is None)
        if missing_count > 0:
            print(f"Warning: {missing_count} nodes missing embeddings. Using zero vectors.")
            embedding_dim = len(next(iter(gene_dict.values())))
            for i in range(len(node_features)):
                if node_features[i] is None:
                    node_features[i] = np.zeros(embedding_dim)

        node_features = torch.tensor(node_features, dtype=torch.float).to(device)
        print(f"Node features shape: {node_features.shape}")
    else:
        print("\n--- Using One-hot Encoding for Node Features ---")
        node_features = torch.eye(num_nodes).to(device)
        print(f"Node features shape: {node_features.shape}")

    # === 8. Split Train/Val Edges ===
    print("\n--- Splitting Train/Val Data ---")
    indices = torch.arange(all_train_val_edges.shape[1])

    # 80/20 split for train/val
    train_indices, val_indices = train_test_split(
        indices.numpy(),
        test_size=0.2,
        stratify=all_train_val_labels.numpy(),
        random_state=42
    )

    train_edge_label_index = all_train_val_edges[:, train_indices]
    train_edge_label = all_train_val_labels[train_indices]
    val_edge_label_index = all_train_val_edges[:, val_indices]
    val_edge_label = all_train_val_labels[val_indices]

    print(f"Training: {len(train_edge_label)} edges ({train_edge_label.sum().item():.0f} pos / {(1-train_edge_label).sum().item():.0f} neg)")
    print(f"Validation: {len(val_edge_label)} edges ({val_edge_label.sum().item():.0f} pos / {(1-val_edge_label).sum().item():.0f} neg)")

    # === 9. Message Passing Graph (use training edges only) ===
    # FIX: Only use positive edges for message passing, do not include negative samples
    train_pos_mask = train_edge_label == 1
    train_message_passing_edge_index = train_edge_label_index[:, train_pos_mask]

    # === 10. Initialize Model and Optimizer ===
    print("\n--- Model Setup ---")
    model = GNNLinkPredictor(node_features.shape[1], 128, 64).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # === 11. Training Functions ===
    def train():
        model.train()
        optimizer.zero_grad()

        # Use message passing graph (training edges only)
        z = model.encode(node_features.to(device), train_message_passing_edge_index.to(device))

        # Predict on training labeled edges
        out = model.decode(z, train_edge_label_index.to(device))
        loss = criterion(out, train_edge_label.to(device))

        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(edge_label_index, edge_label):
        model.eval()

        # Use training message passing graph for encoding
        z = model.encode(node_features.to(device), train_message_passing_edge_index.to(device))

        # Predict on the given edges
        out = model.decode(z, edge_label_index.to(device)).sigmoid()

        labels = edge_label.cpu().numpy()
        preds = out.cpu().numpy()
        return labels, preds

    # === 12. Training Loop ===
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)

    best_val_auc = 0
    best_test_labels = None
    best_test_preds = None
    best_epoch = 0

    for epoch in range(1, 501):
        loss = train()

        # Get labels and predictions for validation and test sets
        val_labels, val_preds = test(val_edge_label_index, val_edge_label)
        test_labels_epoch, test_preds_epoch = test(test_edge_index, test_labels)

        # Calculate AUC scores from the results
        val_auc = roc_auc_score(val_labels, val_preds)
        test_auc = roc_auc_score(test_labels_epoch, test_preds_epoch)

        val_f1 = get_f1_score(val_labels, val_preds)
        test_f1 = get_f1_score(test_labels_epoch, test_preds_epoch)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_labels = test_labels_epoch
            best_test_preds = test_preds_epoch
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_gnn_model.pth')
            print(
                f"Epoch {epoch:03d}: Loss={loss:.4f} | Val: AUC={val_auc:.4f}, F1={val_f1:.4f} | Test: AUC={test_auc:.4f}, F1={test_f1:.4f} ★ NEW BEST (Saved)")
        elif epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d}: Loss={loss:.4f} | Val: AUC={val_auc:.4f}, F1={val_f1:.4f} | Test: AUC={test_auc:.4f}, F1={test_f1:.4f}")

    # === 14. Final Results ===
    print("\n" + "="*80)
    print("Training Complete")
    print("="*80)

    if best_test_labels is not None and best_test_preds is not None:
        final_auc = roc_auc_score(best_test_labels, best_test_preds)
        final_f1 = get_f1_score(best_test_labels, best_test_preds)

        print(f"\nBest model from epoch: {best_epoch}")
        print(f"Best validation AUC: {best_val_auc:.4f}")
        print(f"Final Test AUC: {final_auc:.4f}")
        print(f"Final Test F1: {final_f1:.4f}")
        print(f"Model saved to: best_gnn_model.pth")

        # --- Save Predictions ---
        print("\n--- Saving Test Predictions ---")
        # Create reverse mapping
        idx_to_node = {i: name for name, i in node_to_idx.items()}

        # Get source and target indices from test_edge_index
        src_indices = test_edge_index[0].cpu().numpy()
        tgt_indices = test_edge_index[1].cpu().numpy()

        src_names = [idx_to_node[i] for i in src_indices]
        tgt_names = [idx_to_node[i] for i in tgt_indices]

        pred_df = pd.DataFrame({
            'source': src_names,
            'target': tgt_names,
            'label': best_test_labels,
            'prediction': best_test_preds
        })

        pred_df.to_csv('gnn_test_predictions.csv', index=False)
        print("Predictions saved to: gnn_test_predictions.csv")
    else:
        print("\nNo valid results obtained.")
        final_auc = 0

    print("="*80)

    # === 15. Visualization ===
    if best_test_labels is not None and best_test_preds is not None:
        # Calculate the ROC curve points
        fpr, tpr, thresholds = roc_curve(best_test_labels, best_test_preds)

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {final_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

        # Add labels and title
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - SIGNOR Dataset')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        # Save and show the plot
        plt.tight_layout()
        plt.savefig('signor_roc_curve.png', dpi=300, bbox_inches='tight')
        print(f"\nROC curve saved to: signor_roc_curve.png")
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a GNN for link prediction on SIGNOR dataset. "
                    "Training and validation use full graph with randomly generated negative edges. "
                    "Test set uses high-quality true positive and true negative labels."
    )
    parser.add_argument(
        '--use_text_embeddings',
        action='store_true',
        help="If set, use pre-generated text embeddings as node features. Otherwise use one-hot encoding."
    )
    parser.add_argument(
        '--embedding_path',
        type=str,
        default='data/gene_embeddings.npy',
        help="Path to the .npy file containing node text embeddings (required if --use_text_embeddings is set)."
    )
    args = parser.parse_args()
    main(args)
