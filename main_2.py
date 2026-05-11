import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv # استخدام طبقات Attention
from utils import load_cora_data
import matplotlib.pyplot as plt

# بناء شبكة Graph Attention Network (GAT)
class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        # الطبقة الأولى: 8 رؤوس انتباه (heads)، كل رأس يعالج 8 ميزات
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        # الطبقة الثانية: تجميع النتائج للتصنيف النهائي
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # تطبيق Dropout في البداية لحماية البيانات من الحفظ الصم (Overfitting)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x) # دالة التفعيل ELU مناسبة جداً للـ GAT
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def main():
    print("--- Loading Cora Dataset (Advanced GAT Mode) ---")
    data = load_cora_data()

    # تعريف الموديل والمحسن (Optimizer)
    model = GAT(num_features=data.num_features, num_classes=int(data.y.max()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # قوائم لتخزين النتائج للرسم البياني
    history = {'loss': [], 'acc': []}

    print("\n--- Training Progress ---")
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # حساب الدقة في كل خطوة لمراقبة التطور
        model.eval()
        pred = model(data).argmax(dim=1)
        acc = int((pred[data.train_mask] == data.y[data.train_mask]).sum()) / int(data.train_mask.sum())
        
        history['loss'].append(loss.item())
        history['acc'].append(acc)

        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | Training Acc: {acc:.4f}')

    # --- توليد الرسم البياني ---
    print("\n--- Generating Learning Curves ---")
    plt.figure(figsize=(12, 5))
    
    # رسم الخسارة (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], color='tab:red', linewidth=2)
    plt.title('Model Loss (Convergence)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle='--', alpha=0.6)

    # رسم الدقة (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], color='tab:blue', linewidth=2)
    plt.title('Training Accuracy (Learning Rate)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('learning_curves_2.png')
    print("[Done] Results saved as 'learning_curves.png'")

    # التقييم النهائي (Final Test)
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    final_acc = int(correct) / int(data.test_mask.sum())
    
    print(f'\n' + '='*30)
    print(f'FINAL TEST ACCURACY: {final_acc*100:.2f}%')
    print('='*30)

if __name__ == "__main__":
    main()