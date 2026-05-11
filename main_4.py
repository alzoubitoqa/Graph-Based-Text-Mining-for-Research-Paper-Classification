import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from utils import load_cora_data
import matplotlib.pyplot as plt
import random
import numpy as np
import os


# =========================
# تثبيت العشوائية Seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # لجعل النتائج أكثر ثباتًا قدر الإمكان
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# حساب الدقة Accuracy
# =========================
@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)

    correct = (pred[mask] == data.y[mask]).sum().item()
    total = mask.sum().item()

    return correct / total


# =========================
# بناء موديل GATv2
# =========================
class GATv2(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_channels=16,
        heads=8,
        dropout=0.4
    ):
        super(GATv2, self).__init__()

        self.dropout = dropout

        # الطبقة الأولى من GATv2
        self.conv1 = GATv2Conv(
            in_channels=num_features,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True
        )

        # الطبقة الثانية للتصنيف النهائي
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=num_classes,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Dropout قبل الطبقة الأولى
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GATv2 Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        # Dropout قبل الطبقة الثانية
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GATv2 Layer 2
        x = self.conv2(x, edge_index)

        return x


# =========================
# تدريب موديل واحد
# =========================
def train_one_run(
    data,
    device,
    seed=42,
    hidden_channels=16,
    heads=8,
    dropout=0.4,
    lr=0.005,
    weight_decay=5e-4,
    max_epochs=500,
    patience=100
):
    set_seed(seed)

    model = GATv2(
        num_features=data.num_features,
        num_classes=int(data.y.max()) + 1,
        hidden_channels=hidden_channels,
        heads=heads,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=20
    )

    history = {
        "loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": []
    }

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    best_model_path = f"best_gatv2_seed_{seed}.pt"

    print("\n" + "=" * 60)
    print(f"Training Run with Seed = {seed}")
    print("=" * 60)

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data)

        loss = F.cross_entropy(
            out[data.train_mask],
            data.y[data.train_mask]
        )

        loss.backward()
        optimizer.step()

        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)
        test_acc = evaluate(model, data, data.test_mask)

        scheduler.step(val_acc)

        history["loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        # حفظ أفضل موديل حسب Validation Accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_epoch = epoch
            patience_counter = 0

            torch.save(model.state_dict(), best_model_path)
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

    # تحميل أفضل موديل
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_train_acc = evaluate(model, data, data.train_mask)
    final_val_acc = evaluate(model, data, data.val_mask)
    final_test_acc = evaluate(model, data, data.test_mask)

    print("\n--- Best Model Result for this Seed ---")
    print(f"Seed: {seed}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Final Train Accuracy: {final_train_acc * 100:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc * 100:.2f}%")
    print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")

    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "train_acc": final_train_acc,
        "val_acc": final_val_acc,
        "test_acc": final_test_acc,
        "history": history,
        "model_path": best_model_path
    }

    return result


# =========================
# رسم أفضل تجربة
# =========================
def plot_best_history(history, seed):
    print("\n--- Generating Learning Curves ---")

    plt.figure(figsize=(14, 5))

    # رسم Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], linewidth=2)
    plt.title("Model Loss - GATv2")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)

    # رسم Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    plt.plot(history["val_acc"], label="Validation Accuracy", linewidth=2)
    plt.plot(history["test_acc"], label="Test Accuracy", linewidth=2)
    plt.title("Accuracy Curves - GATv2")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    output_name = f"learning_curves_gatv2_seed_{seed}.png"
    plt.savefig(output_name)

    print(f"[Done] Results saved as '{output_name}'")


# =========================
# البرنامج الرئيسي
# =========================
def main():
    print("--- Loading Cora Dataset Advanced GATv2 Mode ---")

    data = load_cora_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    print(f"Using device: {device}")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes: {int(data.y.max()) + 1}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.size(1)}")

    # جرّبي عدة Seeds لأن GAT حساس للعشوائية
    seeds = [42, 7, 123, 2024, 2026]

    all_results = []

    for seed in seeds:
        result = train_one_run(
            data=data,
            device=device,
            seed=seed,
            hidden_channels=16,
            heads=8,
            dropout=0.4,
            lr=0.005,
            weight_decay=5e-4,
            max_epochs=500,
            patience=100
        )

        all_results.append(result)

    # اختيار أفضل نتيجة حسب Validation Accuracy
    best_result = max(all_results, key=lambda x: x["val_acc"])

    print("\n" + "=" * 70)
    print("FINAL SUMMARY FOR ALL SEEDS")
    print("=" * 70)

    for result in all_results:
        print(
            f"Seed: {result['seed']} | "
            f"Best Epoch: {result['best_epoch']} | "
            f"Train: {result['train_acc'] * 100:.2f}% | "
            f"Val: {result['val_acc'] * 100:.2f}% | "
            f"Test: {result['test_acc'] * 100:.2f}%"
        )

    print("\n" + "=" * 70)
    print("BEST OVERALL MODEL")
    print("=" * 70)

    print(f"Best Seed: {best_result['seed']}")
    print(f"Best Epoch: {best_result['best_epoch']}")
    print(f"Best Train Accuracy: {best_result['train_acc'] * 100:.2f}%")
    print(f"Best Validation Accuracy: {best_result['val_acc'] * 100:.2f}%")
    print(f"Best Test Accuracy: {best_result['test_acc'] * 100:.2f}%")
    print(f"Best Model Saved As: {best_result['model_path']}")

    # رسم أفضل تجربة
    plot_best_history(
        history=best_result["history"],
        seed=best_result["seed"]
    )


if __name__ == "__main__":
    main()