import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from dgl.nn import GATConv

torch.manual_seed(25)
np.random.seed(25)
dgl.seed(25)


class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, num_heads=4, dropout=0.5):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(
            in_feats, hidden_size, num_heads=num_heads,
            feat_drop=dropout, attn_drop=dropout, activation=F.elu
        )
        self.conv2 = GATConv(
            hidden_size * num_heads, num_classes, num_heads=1,
            feat_drop=dropout, attn_drop=dropout, activation=None
        )
        self.dropout = dropout

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = h.flatten(1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(g, h)
        h = h.mean(1)
        return h


def load_data():
    print("Loading data...")

    # ─────────────────────────────────────────────────────────────
    # CHOOSE YOUR DATA FORMAT:
    #   "free" → train_graph_free.pkl  (no DGL needed to load data)
    #   "dgl"  → train_graph.pkl       (requires DGL installed)
    DATA_FORMAT = "free"   # ← change to "dgl" if you prefer
    # ─────────────────────────────────────────────────────────────

    if DATA_FORMAT == "free":
        with open('../data/public/train_graph_free.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('../data/public/test_graph_free.pkl', 'rb') as f:
            test_data = pickle.load(f)

        def rebuild_dgl_graph(d):
            src, dst = d["edge_index"]
            g = dgl.graph((src, dst), num_nodes=d["num_nodes"])
            d["graph"] = g
            return d

        train_data = rebuild_dgl_graph(train_data)
        test_data  = rebuild_dgl_graph(test_data)
        print("  Loaded DGL-free format (rebuilt graph at runtime)")

    elif DATA_FORMAT == "dgl":
        with open('../data/public/train_graph.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('../data/public/test_graph.pkl', 'rb') as f:
            test_data = pickle.load(f)
        print("  Loaded DGL format")

    else:
        raise ValueError(f"Unknown DATA_FORMAT '{DATA_FORMAT}'. Choose 'free' or 'dgl'.")

    return train_data, test_data


def train_epoch(model, g, features, labels, train_mask, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()
    logits = model(g, features)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(logits[train_mask], 1)
    train_acc = (predicted == labels[train_mask]).float().mean()
    return loss.item(), train_acc.item()


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        _, predicted = torch.max(logits[mask], 1)
        accuracy = (predicted == labels[mask]).float().mean()
        f1_macro = f1_score(
            labels[mask].cpu().numpy(),
            predicted.cpu().numpy(),
            average='macro'
        )
    return accuracy.item(), f1_macro


def main():
    print("=" * 60)
    print("GNN Parkinson's Challenge - Baseline GAT Model")
    print("=" * 60)

    train_data, test_data = load_data()

    g          = train_data['graph']
    features   = train_data['features']
    labels     = train_data['labels']
    train_mask = train_data['train_mask']
    val_mask   = train_data['val_mask']

    print(f"\nDataset Statistics:")
    print(f"  Nodes: {g.num_nodes()}")
    print(f"  Edges: {g.num_edges()}")

    # ✅ Fix class imbalance with weighted loss
    train_labels = labels[train_mask]
    num_class0 = (train_labels == 0).sum().item()
    num_class1 = (train_labels == 1).sum().item()
    total = num_class0 + num_class1
    w0 = total / (2 * num_class0)
    w1 = total / (2 * num_class1)
    class_weights = torch.FloatTensor([w0, w1])
    print(f"  Class weights: Healthy={w0:.2f}, Parkinson's={w1:.2f}")

    in_feats     = features.shape[1]
    hidden_size  = 32
    num_classes  = 2
    num_heads    = 4
    dropout      = 0.6
    lr           = 0.005
    weight_decay = 5e-4
    num_epochs   = 250

    print(f"\nModel: GAT with {num_heads} attention heads")

    model     = GATModel(in_feats, hidden_size, num_classes, num_heads, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("\nTraining...")
    print("-" * 60)

    best_val_f1      = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        loss, train_acc = train_epoch(model, g, features, labels, train_mask, optimizer, class_weights)
        val_acc, val_f1 = evaluate(model, g, features, labels, val_mask)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_gat_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= 50:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nBest Validation F1: {best_val_f1:.4f}")

    model.load_state_dict(torch.load('best_gat_model.pt'))

    print("\nGenerating predictions...")
    test_g        = test_data['graph']
    test_features = test_data['features']
    test_node_ids = test_data['node_ids']  # numpy array of 39 node indices

    model.eval()
    with torch.no_grad():
        test_logits = model(test_g, test_features)
        # ✅ Only extract predictions for the 39 test node IDs
        test_predictions = torch.max(test_logits[test_node_ids], 1)[1]

    import os
    os.makedirs('../submissions', exist_ok=True)
    submission = pd.DataFrame({
        'node_id':    test_node_ids,
        'prediction': test_predictions.cpu().numpy()
    })

    submission.to_csv('../submissions/gat_submission.csv', index=False)
    print(f"Submission saved! {len(submission)} predictions.")
    print(submission.head(10))
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()