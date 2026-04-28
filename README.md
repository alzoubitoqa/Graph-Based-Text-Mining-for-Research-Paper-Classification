# 📚 Graph-Based Text Mining for Research Paper Classification

## 👩‍💻 Authors
- Toqa Al-Zoubi  
- Shaden Alshobake  
- Ghaida Breitem  

## 👨‍🏫 Supervisor
Prof. Abdelwadood Mesleh  

---

## 📌 Project Overview
This project focuses on classifying research papers into **7 categories** using both:
- Textual content (Bag-of-Words)
- Citation relationships (Graph structure)

Dataset used:
- **Cora Citation Network**

---

## 🎯 Objective
To build a deep learning model that improves classification accuracy by combining:
- Natural Language Processing
- Graph-based learning

---

## 🧠 Methodology

### 1. Text Mining
- Bag-of-Words representation (1433 features)

### 2. Graph Representation
- Papers → Nodes  
- Citations → Edges  

### 3. Model
- Graph Convolutional Network (GCN)

---

## 🏗️ Model Architecture

- Input Layer: (2708 × 1433)
- GCN Layer 1:
  - 16 hidden units
  - ReLU activation
  - Dropout = 0.5
- GCN Layer 2:
  - Output = 7 classes
- Output Layer:
  - Log Softmax

---

## 📊 Results

- ✅ Accuracy: **80.50%**
- 📉 Loss: from 0.2652 → 0.0362
- 🔍 Key Insight:
  > Citation relationships significantly improve classification performance.

---

## ⚙️ Technologies Used

- Python 3.10
- PyTorch
- Torch Geometric
- Pandas

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/alzoubitoqa/Graph-Based-Text-Mining-for-Research-Paper-Classification.git

# 2. Go to project
cd Graph-Based-Text-Mining-for-Research-Paper-Classification

# 3. Install requirements
pip install -r requirements.txt

# 4. Run the model
python main.py

📂 Project Structure
├── data/
├── main.py
├── requirements.txt
└── README.md
🧪 Dataset

Cora Citation Network Dataset

📈 Conclusion

This project demonstrates that combining graph structure with text data significantly enhances classification accuracy in research paper categorization.