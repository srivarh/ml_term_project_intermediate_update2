

# **Disaster Tweet Classification Using Deep Learning**

### *Machine Learning Term Project – Final Submission*

### *Mahitha Sri Varshini Alla – TAMUSA*

---

## 📌 **Project Overview**

This project aims to automatically classify tweets as **disaster-related** or **non-disaster** using deep learning techniques. Using the Kaggle dataset “Real or Not? Disaster Tweets,” we build a complete NLP pipeline that includes preprocessing, tokenization, sequence modeling, and performance evaluation across multiple architectures (LSTM, Tuned LSTM, GRU).

The final goal:
**Determine which model performs best for disaster tweet classification.**

---

## 📂 **Repository Structure**

```
project/
│
├── data/
│   └── train.csv (not included due to size restrictions)
│
├── notebooks/
│   ├── WEEK2_BASELINE.ipynb
│   └── WEEK3_TUNING_AND_GRU.ipynb
│
├── models/
│   ├── best_gru_model.h5   (may not be uploaded due to size limits)
│
├── ML_paper.pdf   (IEEE-format project paper)
└── README.md       (this file)
```

---

# 🧭 **1. Dataset Description**

**Dataset:** “Real or Not? Disaster Tweets” from Kaggle
**Total Tweets:** 7,613
**Label Meaning:**

* `1` → Disaster-related tweet
* `0` → Not a disaster tweet

**Fields:** text, keyword, location, target

The dataset is short-text, noisy, and partially imbalanced, making it ideal for LSTM/GRU sequence models.

---

# 🧹 **2. Preprocessing Pipeline**

The following steps were applied:

### ✔ Text cleaning

* Remove URLs
* Remove mentions (@user)
* Remove hashtags
* Remove punctuation
* Lowercase text
* Remove extra spaces

### ✔ Tokenization

* Used Keras Tokenizer
* Vocab size: MAX_VOCAB
* Convert words → integer sequences

### ✔ Padding

* All sequences padded to **length 64** for uniform model input

### ✔ Train/Validation Split

* **80% training**
* **20% validation**

---

# 🧠 **3. Model Architectures**

We experimented with three models:

---

## 🔹 **3.1 Baseline LSTM**

* Embedding: 128
* BiLSTM: 64 units
* Dense Layer: 64 units (ReLU)
* Dropout: 0.4
* Sigmoid output

---

## 🔹 **3.2 Tuned LSTM (Regularized)**

* LSTM units reduced: 64 → 48
* Increased dropout: 0.4 → 0.5
* Purpose: reduce overfitting

---

## 🔹 **3.3 GRU Model (Final Best Model)**

* GRU units: 64
* Dropout: 0.4
* Dense Layer: 64 units
* Faster + more stable than LSTM

---

# 🧪 **4. Training Configuration**

* Loss function: **Binary Crossentropy**
* Optimizer: **Adam**
* Batch size: **64**
* Epochs: **5–10**
* EarlyStopping enabled
* Metrics: Accuracy + Precision + Recall + F1

All models were trained using a unified function `train_and_evaluate()` for fair comparison.

---

# 📈 **5. Experiment Results**

## ✔ **5.1 Baseline LSTM**

* **Val Accuracy:** 0.803
* **Macro F1:** 0.794
* Mild overfitting observed

---

## ✔ **5.2 Tuned LSTM**

* **Val Accuracy:** 0.798
* **Macro F1:** 0.788
* Better generalization but slightly worse performance

---

## ✔ **5.3 GRU Model (Best)**

* **Val Accuracy:** 0.820
* **Macro F1:** 0.812
* Fastest convergence
* Most stable validation curves
* Selected as final model

---

# 📊 **6. Final Model Comparison**

| Model         | Val Accuracy | Macro F1  |
| ------------- | ------------ | --------- |
| Baseline LSTM | 0.803        | 0.794     |
| Tuned LSTM    | 0.798        | 0.788     |
| **GRU Model** | **0.820**    | **0.812** |

**Conclusion:**
➡️ **GRU outperformed both LSTM models** and is the selected final architecture.

---

# 🔍 **7. Confusion Matrix (GRU Model)**

|          | Pred 0  | Pred 1  |
| -------- | ------- | ------- |
| Actual 0 | **774** | 95      |
| Actual 1 | 179     | **475** |

Insights:

* High true positives + high true negatives
* Balanced performance
* Fewer false alarms
* Fewer missed disasters

---

# 💬 **8. Discussion**

### **Strengths**

* GRU handled short text extremely well
* Faster training than LSTM
* Strong generalization ability
* F1 score improved over baseline

### **Observations**

* LSTM models tend to overfit faster
* Regularization may reduce performance
* GRU provided the best balance

---

# ⚠️ **9. Limitations**

* Dataset is small
* Sarcasm and ambiguous text remain difficult
* English-only tweets cause language bias
* Missing metadata (location, keyword) limits context
* Overfitting risk due to limited data

---

# 🧭 **10. Ethical Considerations**

* Misclassification can impact emergency response
* False negatives might ignore real disasters
* False positives could create panic
* Model must not be used without human oversight
* Bias toward certain regions or writing styles

---

# 🚀 **11. Future Work**

* Add BERT / RoBERTa transformer models
* Use keyword and location metadata
* Build real-time streaming pipeline (Kafka, Pub/Sub)
* Apply explainable AI (SHAP, LIME)
* Expand dataset with multilingual tweets

---

# 🛠 **12. How to Run This Project**

### **Step 1: Install dependencies**

```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib nltk
```

### **Step 2: Open Notebook**

```bash
jupyter notebook WEEK3_TUNING_AND_GRU.ipynb
```

### **Step 3: Run all cells**

This will:

* Load dataset
* Preprocess tweets
* Train LSTM and GRU models
* Display metrics + confusion matrix

---

# 👨‍🏫 **13. Instructor Access**

Professor **GB-TonyLiang** has been added as a read-only GitHub collaborator.

---

# 🎉 **14. Final Summary**

This project successfully built a complete NLP pipeline for disaster tweet classification. Multiple models were trained and evaluated, and the **GRU model** achieved the best performance with a macro F1-score of **0.812**.
All results, experiments, and comparisons show that deep learning is highly effective for short-text disaster classification.

---
