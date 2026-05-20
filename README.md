# AI-First Fraud and Risk Management in U.S. Digital Banking

**Bhavesh Kulkarni** · Northeastern University, College of Engineering — Data Analytics Engineering

---

## Overview

This project develops a deep-learning baseline fraud detection model using a **Multi-Layer Perceptron (MLP)** trained on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection). Beyond standard classification metrics, the work situates model performance within a practical U.S. banking risk-management context — addressing threshold selection, expected-dollar-loss optimization, calibration, and operational trade-offs.

**Key result:** ROC-AUC of **0.8649** with threshold-tuned recall and calibrated probability outputs suitable for risk-tiering.

---

## Repository Structure

```
├── notebooks/
│   ├── EDA_IEEE_CIS_Fraud_Detection.ipynb   # Exploratory data analysis
│   └── MLP_Fraud_Detection_Model.ipynb      # Model training, evaluation & analysis
├── reports/
│   └── Bhavesh_Kulkarni_MS_Project_Report.pdf  # Full project report
├── paper/
│   └── Bhavesh_Kulkarni_Short_Paper.pdf     # IEEE-style short paper
└── README.md
```

---

## Dataset

The **IEEE-CIS Fraud Detection** dataset is available on Kaggle:  
[https://www.kaggle.com/c/ieee-fraud-detection/data](https://www.kaggle.com/c/ieee-fraud-detection/data)

- **590,540** training transactions with **434** engineered features
- Target variable `isFraud` — ~**0.3% fraud prevalence** (severely imbalanced)
- Features include transaction attributes (V\*, C\*, D\*) and identity fields (id\_\*)
- Real-world challenges: high missingness (70% of features), temporal drift, anonymization

> The dataset files are **not included** in this repository due to size. Download from Kaggle and place `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, and `test_identity.csv` in a `data/` directory.

---

## Methodology

### Exploratory Data Analysis
- Missing value analysis across 434 features
- Correlation structure within V\*, C\*, D\*, and id\_\* feature groups
- Temporal drift analysis between train and test distributions
- Class imbalance visualization

### Model Architecture (MLP)

```
Input (186 features)
  → Dense(50, ReLU)
  → Dense(25, ReLU)
  → Dense(12, ReLU)
  → Dropout(0.5)
  → BatchNormalization
  → Dense(1, Sigmoid)
```

- **10,998 trainable parameters** — lightweight baseline
- Optimizer: Adam | Loss: Binary Cross-Entropy
- 5 epochs, batch size 64, early stopping

### Evaluation Framework
- ROC-AUC and Precision-Recall curves
- Confusion matrix at multiple thresholds
- Threshold sensitivity analysis (0.0 – 0.7)
- Expected-dollar-loss cost model
- Calibration curve (reliability diagram)

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.8649** |
| Average Precision (PR-AUC) | 0.4257 |
| Precision @ threshold=0.50 | 0.8585 |
| Recall @ threshold=0.50 | 0.1805 |
| Brier Score (calibration) | 0.0258 |
| Optimal expected-loss threshold | **0.10** |

**Key findings:**
- Default threshold (0.50) yields high precision but very low recall — not suitable for fraud operations
- Lowering threshold to **0.10** minimizes expected financial loss, significantly increasing fraud capture
- F1-score peaks in the 0.1–0.3 threshold range
- Calibration curve confirms probabilities align with true fraud likelihood, supporting risk-tiering

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow networkx nodevectors tqdm
```

Python 3.10+

---

## Key Insights for Banking Operations

1. **Optimize for expected loss, not accuracy.** With 0.3% fraud prevalence, accuracy is meaningless — a model predicting all-legitimate achieves 99.7% accuracy.
2. **Threshold is a business decision.** The optimal threshold depends on the cost ratio of false negatives (missed fraud) vs. false positives (false alerts).
3. **Calibrated probabilities enable risk-tiering.** Well-calibrated scores let institutions route transactions to different review queues based on estimated loss exposure.
4. **This baseline is not production-ready alone.** It should be augmented with rules, anomaly detection, and graph-based features for real deployment.

---

## Future Work

- Cost-sensitive loss functions and class-weighted training
- Graph Neural Networks (GNNs) for fraud ring / mule network detection
- LSTM / Transformer models for temporal sequence modeling
- SHAP-based explainability for SR 11-7 / FFIEC compliance
- Drift monitoring and real-time scoring pipeline integration
- Hybrid architecture: supervised ML + anomaly detection

---

## References

1. Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, pp. 436–444, 2015.
2. TensorFlow: An End-to-End Open Source Machine Learning Platform.
3. IEEE Computational Intelligence Society — IEEE-CIS Fraud Detection Challenge.
4. Python Software Foundation, "Python 3.10 documentation," 2025.

---

## License

This project is for academic and educational purposes as part of an M.S. program at Northeastern University.
