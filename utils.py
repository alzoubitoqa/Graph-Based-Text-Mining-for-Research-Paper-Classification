import pandas as pd
import torch
from torch_geometric.data import Data

def load_cora_data(folder_path='data/'):
    # 1. تحميل ميزات النصوص (Nodes Features)
    # كل صف يمثل ورقة بحثية وكل عمود (1433) يمثل كلمة
    x_df = pd.read_csv(f'{folder_path}x.csv')
    x = torch.tensor(x_df.values, dtype=torch.float)

    # 2. تحميل روابط الاقتباس (Edges)
    edge_df = pd.read_csv(f'{folder_path}edge_index.csv')
    # الـ GNN يحتاج الحواف بصيغة [2, number_of_edges]
    edge_index = torch.tensor(edge_df.values.T, dtype=torch.long)

    # 3. تحميل التصنيفات (Labels) وتجهيز الـ Masks
    y_train = pd.read_csv(f'{folder_path}y_train.csv')
    y_val = pd.read_csv(f'{folder_path}y_val.csv')
    y_test = pd.read_csv(f'{folder_path}y_test.csv')
    
    num_nodes = x.shape[0]
    y = torch.zeros(num_nodes, dtype=torch.long)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # تعبئة القيم وتحديد الأقنعة (Masks)
    for df, mask in [(y_train, train_mask), (y_val, val_mask), (y_test, test_mask)]:
        for _, row in df.iterrows():
            idx = int(row['index'])
            y[idx] = int(row['label'])
            mask[idx] = True

    # تجميع البيانات في كائن واحد
    data = Data(x=x, edge_index=edge_index, y=y, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    return data