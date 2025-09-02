import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def main(args):
    # --- Load and Preprocess Data ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_path = 'data/corrected_PKN_with_BRAF_fixes.csv'
    df = pd.read_csv(file_path)

    all_nodes = pd.unique(df[['source', 'target']].values.ravel())
    node_to_idx = {name: i for i, name in enumerate(all_nodes)}
    num_nodes = len(all_nodes)

    source_nodes = [node_to_idx[name] for name in df['source']]
    target_nodes = [node_to_idx[name] for name in df['target']]

    positive_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # --- Feature Generation based on args ---
    if args.use_text_embeddings:
        print(f"\n--- Attempting to load node features from text embeddings: {args.embedding_path} ---")
        node_features = [None for _ in range(num_nodes)]
        gene_dict = np.load(args.embedding_path, allow_pickle=True).item()
        for gene, embedding in gene_dict.items():
            node_features[node_to_idx[gene]] = embedding

        node_features = torch.tensor(node_features, dtype=torch.float).to(device)
    else:
        print("\n--- Initializing node features with one-hot encoding ---")
        node_features = torch.eye(num_nodes).to(device)

    print(f"\nOriginal graph has {num_nodes} nodes and {positive_edge_index.shape[1]} edges.")

    # --- Perturb the Graph with Negative Edges ---
    print("Perturbing graph with random negative edges...")
    existing_edges = set(tuple(e) for e in positive_edge_index.t().tolist())
    num_negative_edges = positive_edge_index.shape[1]
    negative_edges_list = []
    negative_edges_set = set()
    while len(negative_edges_list) < num_negative_edges:
        u, v = np.random.randint(0, num_nodes, 2)
        if u != v and (u, v) not in existing_edges and (u, v) not in negative_edges_set:
            negative_edges_list.append([u, v])
            negative_edges_set.add((u, v))

    negative_edge_index = torch.tensor(negative_edges_list).t().contiguous()
    print(f"Added {negative_edge_index.shape[1]} negative edges.")

    # --- Create Labels and Combine Edges ---
    positive_labels = torch.ones(positive_edge_index.shape[1])
    negative_labels = torch.zeros(negative_edge_index.shape[1])
    all_edges = torch.cat([positive_edge_index, negative_edge_index], dim=1)
    all_labels = torch.cat([positive_labels, negative_labels], dim=0)

    # --- Split Edges into Train, Validation, and Test Sets ---
    indices = torch.arange(all_edges.shape[1])
    train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=all_labels)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.25, stratify=all_labels[train_indices])

    train_edge_label_index = all_edges[:, train_indices]
    train_edge_label = all_labels[train_indices]
    val_edge_label_index = all_edges[:, val_indices]
    val_edge_label = all_labels[val_indices]
    test_edge_label_index = all_edges[:, test_indices]
    test_edge_label = all_labels[test_indices]

    # --- Message Passing Graph---
    train_message_passing_edge_index = train_edge_label_index

    print("\n--- Data after Splitting ---")
    print(
        f"Message-passing graph edges: {train_message_passing_edge_index.shape[1]} (now includes positive AND negative edges)")
    print(f"Training supervision edges: {len(train_edge_label)}")
    print(f"Validation supervision edges: {len(val_edge_label)}")
    print(f"Test supervision edges: {len(test_edge_label)}")
    print("----------------------------\n")

    model = GNNLinkPredictor(node_features.shape[1], 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    def train():
        model.train()
        optimizer.zero_grad()

        z = model.encode(node_features.to(device), train_message_passing_edge_index.to(device))
        out = model.decode(z, train_edge_label_index.to(device))
        loss = criterion(out, train_edge_label.to(device))
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(edge_label_index, edge_label):
        model.eval()
        # Note: The model encodes using the full, noisy training graph to generate embeddings for evaluation
        # z = model.encode(node_features.to(device), train_message_passing_edge_index.to(device))
        z = model.encode(node_features.to(device), edge_label_index.to(device))
        out = model.decode(z, edge_label_index.to(device)).sigmoid()
        labels = edge_label.cpu().numpy()
        preds = out.cpu().numpy()
        return labels, preds

    print("--- Starting Model Training (on full, perturbed graph) ---")
    best_val_auc = 0
    best_test_labels = None
    best_test_preds = None
    for epoch in range(1, 201):
        loss = train()

        # Get labels and predictions for validation and test sets
        val_labels, val_preds = test(val_edge_label_index, val_edge_label)
        test_labels, test_preds = test(test_edge_label_index, test_edge_label)

        # Calculate AUC scores from the results
        val_auc = roc_auc_score(val_labels, val_preds)
        test_auc = roc_auc_score(test_labels, test_preds)

        val_f1 = get_f1_score(val_labels, val_preds)
        test_f1 = get_f1_score(test_labels, test_preds)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_labels = test_labels
            best_test_preds = test_preds
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_auc:.4f}, Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f} (New Best)")
        elif epoch % 20 == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_auc:.4f}, Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}")

    print("\n--- Training Finished ---")
    final_auc = roc_auc_score(best_test_labels, best_test_preds) if best_test_labels is not None else 0
    print(f"\nFinal Test AUC: {final_auc:.4f}")
    print("-------------------------")

    if best_test_labels is not None and best_test_preds is not None:
        # Calculate the ROC curve points
        fpr, tpr, thresholds = roc_curve(best_test_labels, best_test_preds)

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {final_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No-Skill Classifier')

        # Add labels and title
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        # Show the plot
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GNN for link prediction with optional text embeddings.")
    parser.add_argument(
        '--use_text_embeddings',
        action='store_true',
        help="If set, use pre-generated text embeddings as node features."
    )
    parser.add_argument(
        '--embedding_path',
        type=str,
        default='data/gene_embeddings.npy',
        help="Path to the .npy file containing node text embeddings."
    )
    args = parser.parse_args()
    main(args)