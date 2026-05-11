import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from utils import load_cora_data
import matplotlib.pyplot as plt
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=8, heads=8, dropout=0.5):
        super(GAT, self).__init__()

        self.dropout = dropout

        self.conv1 = GATConv(
            in_channels=num_features,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout
        )

        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=num_classes,
            heads=1,
            concat=False,
            dropout=dropout
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return x


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    total = mask.sum().item()
    return correct / total


def main():
    set_seed(42)

    print("--- Loading Cora Dataset Improved GAT Mode ---")
    data = load_cora_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GAT(
        num_features=data.num_features,
        num_classes=int(data.y.max()) + 1,
        hidden_channels=8,
        heads=8,
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=20
    )

    history = {
        'loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0
    patience = 60
    patience_counter = 0

    print("\n--- Training Progress ---")

    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()

        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)
        test_acc = evaluate(model, data, data.test_mask)

        scheduler.step(val_acc)

        history['loss'].append(loss.item())
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), "best_gat_model.pt")
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print("\n--- Loading Best Model ---")
    model.load_state_dict(torch.load("best_gat_model.pt"))

    final_train_acc = evaluate(model, data, data.train_mask)
    final_val_acc = evaluate(model, data, data.val_mask)
    final_test_acc = evaluate(model, data, data.test_mask)

    print("\n" + "=" * 40)
    print(f"BEST EPOCH: {best_epoch}")
    print(f"FINAL TRAIN ACCURACY: {final_train_acc * 100:.2f}%")
    print(f"FINAL VALIDATION ACCURACY: {final_val_acc * 100:.2f}%")
    print(f"FINAL TEST ACCURACY: {final_test_acc * 100:.2f}%")
    print("=" * 40)

    print("\n--- Generating Learning Curves ---")

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('learning_curves_improved.png')
    print("[Done] Results saved as 'learning_curves_improved.png'")


if __name__ == "__main__":
    main()