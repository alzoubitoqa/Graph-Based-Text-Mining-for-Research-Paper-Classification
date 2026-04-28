import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import load_cora_data

# تعريف بنية الشبكة العصبية (GCN)
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        # الطبقة الأولى: تجميع المعلومات من الجيران
        self.conv1 = GCNConv(num_features, 16)
        # الطبقة الثانية: التصنيف النهائي
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # تمرير عبر الطبقة الأولى + تفعيل ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # الطبقة الأخيرة
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def main():
    # تحميل البيانات باستخدام الدالة من utils.py
    print("--- Loading Dataset ---")
    data = load_cora_data()
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")

    # إعداد النموذج والمحسن (Optimizer)
    model = GCN(num_features=data.num_features, num_classes=int(data.y.max()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # حلقة التدريب (Training Loop)
    print("\n--- Starting Training ---")
    model.train()
    for epoch in range(1, 201):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

    # التقييم (Evaluation)
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    
    # حساب الدقة على بيانات الاختبار
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'\n--- Evaluation ---')
    print(f'Test Accuracy: {acc:.4f}')

if __name__ == "__main__":
    main()