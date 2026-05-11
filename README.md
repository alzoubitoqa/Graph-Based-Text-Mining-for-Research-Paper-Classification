# 📚 Graph-Based Text Mining for Research Paper Classification

## 👩‍💻 Authors
- Toqa Al-Zoubi  

## 👨‍🏫 Supervisor
Prof. Abdelwadood Mesleh  

---

## 📌 Project Overview

This project focuses on classifying research papers into **7 academic categories** using the **Cora Citation Network** dataset.

The system combines two important sources of information:

- **Textual content** using Bag-of-Words features
- **Citation relationships** using graph structure

Instead of treating each research paper as an independent sample, the project represents the dataset as a graph. Each paper is represented as a node, and citation links between papers are represented as edges. This allows the model to learn from both the content of each paper and its neighboring papers in the citation network.

---

## 🎯 Objective

The objective of this project is to build and evaluate graph-based deep learning models for research paper classification.

The project aims to improve classification accuracy by combining:

- Text mining
- Citation network analysis
- Graph Neural Networks
- Attention-based graph learning

---

## 🧪 Dataset

**Dataset:** Cora Citation Network

The dataset contains:

| Item | Value |
|---|---:|
| Number of papers / nodes | 2708 |
| Number of text features | 1433 |
| Number of classes | 7 |
| Number of citation edges | 10556 |

Each paper is represented using a Bag-of-Words feature vector, and citation links are used to build the graph structure.

---

## 🧠 Methodology

### 1. Text Mining

The textual content of each paper is represented using a Bag-of-Words feature vector with **1433 features**.

### 2. Graph Representation

The Cora dataset is modeled as a graph:

- Papers → Nodes  
- Citations → Edges  

This graph representation allows the model to capture relationships between connected papers.

### 3. Graph Neural Networks

Several graph-based deep learning models were implemented and compared:

| Model | Description |
|---|---|
| GCN | Uses graph convolution and neighborhood aggregation |
| GAT | Uses attention mechanisms to assign different importance to neighboring nodes |
| GATv2 | Advanced attention-based graph neural network variant |

---

## 🏗️ Model Architecture

### Baseline GCN

- Input Layer: `2708 × 1433`
- GCN Layer 1:
  - 16 hidden units
  - ReLU activation
  - Dropout = 0.5
- GCN Layer 2:
  - Output = 7 classes
- Output Layer:
  - Log Softmax

### Improved GAT

The best-performing model was the **Improved Graph Attention Network (GAT)**.

It used:

- Multi-head attention
- Dropout regularization
- Weight decay
- Early stopping
- Validation-based model selection
- Learning rate scheduling

---

## 📊 Results

Several experiments were conducted to compare model performance.

| Experiment | Model | Main Improvement | Test Accuracy |
|---|---|---|---:|
| Baseline GCN | GCN | Basic graph convolution | 80.50% |
| Baseline GAT | GAT | Attention-based neighborhood learning | 79.85% |
| Improved GAT | GAT | Early stopping + validation-based model selection + LR scheduler | **82.50%** |
| GATv2 Multi-Seed | GATv2 | Advanced attention + multiple seed testing | 82.10% |

---

## 🏆 Final Selected Model

The final selected model is:

**Improved Graph Attention Network (GAT)**

It achieved the best performance:

| Metric | Result |
|---|---:|
| Best Epoch | 16 |
| Final Train Accuracy | 97.86% |
| Final Validation Accuracy | 80.80% |
| Final Test Accuracy | **82.50%** |

The improved GAT model was selected because it achieved the highest test accuracy among all tested models.

---

## 🔬 GATv2 Multi-Seed Experiment

GATv2 was tested using multiple random seeds to evaluate training stability.

| Seed | Best Epoch | Train Accuracy | Validation Accuracy | Test Accuracy |
|---:|---:|---:|---:|---:|
| 42 | 17 | 100.00% | 80.00% | 81.20% |
| 7 | 8 | 98.57% | 79.40% | 79.80% |
| 123 | 16 | 99.29% | 81.20% | **82.10%** |
| 2024 | 8 | 97.14% | 76.40% | 78.30% |
| 2026 | 5 | 96.43% | 78.60% | 79.30% |

Although GATv2 achieved strong performance, it did not outperform the improved GAT model.

---

## 📈 Learning Curves

The project includes learning curve visualizations generated during training, such as:

- `learning_curves_improved.png`
- `learning_curves_gatv2_seed_123.png`

These curves show the model loss, training accuracy, validation accuracy, and test accuracy during training.

---

## ⚙️ Technologies Used

- Python 3.10
- PyTorch
- Torch Geometric
- Pandas
- NumPy
- Matplotlib

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/alzoubitoqa/Graph-Based-Text-Mining-for-Research-Paper-Classification.git

# 2. Go to project directory
cd Graph-Based-Text-Mining-for-Research-Paper-Classification

# 3. Install requirements
pip install -r requirements.txt

# 4. Run baseline experiment
python main_1.py

# 5. Run second experiment
python main_2.py

# 6. Run improved GAT model
python main_3.py

# 7. Run GATv2 multi-seed experiment
python main_4.

📂 Project Structure
├── data/
├── main_1.py
├── main_2.py
├── main_3.py
├── main_4.py
├── utils.py
├── requirements.txt
├── README.md
├── best_gat_model.pt
├── best_gatv2_seed_123.pt
├── learning_curves_improved.png
└── learning_curves_gatv2_seed_123.png
📌 Key Insight

Citation relationships provide important relational information that text features alone may not capture. By combining textual features with graph structure, Graph Neural Networks can improve research paper classification performance.

The improved GAT model achieved the best result because its attention mechanism allowed the model to focus on the most relevant neighboring papers in the citation graph.

📈 Conclusion

This project demonstrates the effectiveness of Graph Neural Networks in structured text mining tasks. By integrating Bag-of-Words text features with citation network relationships, the system achieved strong classification performance on the Cora Citation Network dataset.

The best result was achieved by the improved GAT model with a final test accuracy of 82.50%.
