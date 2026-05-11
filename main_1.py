import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import load_cora_data
import matplotlib.pyplot as plt

# تعريف بنية الشبكة العصبية (GCN)
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def main():
    print("--- Loading Dataset ---")
    data = load_cora_data()

    model = GCN(num_features=data.num_features, num_classes=int(data.y.max()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # قوائم لتخزين النتائج من أجل الرسم
    losses = []
    train_accs = []

    print("\n--- Starting Training ---")
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # حساب الدقة للتدريب في كل خطوة
        model.eval()
        pred = model(data).argmax(dim=1)
        acc = int((pred[data.train_mask] == data.y[data.train_mask]).sum()) / int(data.train_mask.sum())
        
        losses.append(loss.item())
        train_accs.append(acc)

        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}')

    # --- الجزء الخاص بالرسم البياني ---
    plt.figure(figsize=(10, 5))
    
    # رسم الـ Loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # رسم الـ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy', color='blue')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves_1.png') # حفظ الصورة
    print("\n[Success] Learning curves saved as 'learning_curves.png'")

    # التقييم النهائي على Test Set
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    final_acc = int(correct) / int(data.test_mask.sum())
    print(f'\nFinal Test Accuracy: {final_acc:.4f}')

if __name__ == "__main__":
    main()