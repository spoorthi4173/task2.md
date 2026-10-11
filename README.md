# Loan Eligibility Prediction using Logistic Regression

**Student:** spoorthi4173

**Project Description:**
This project predicts whether a person is eligible for a loan based on their CIBIL Credit Score and Loan Amount Required. It uses Logistic Regression for binary classification (Eligible / Not Eligible). The model is trained on `trades_orderbook_clean_answer_key.csv` (2000 records) and achieves 90.75% accuracy. It also takes live user input and visualizes the decision boundary.

**Files:**
- `eligibility.py` - Main code to load data, train model, predict and plot
- `trades_orderbook_clean_answer_key.csv` - Dataset

**Output:** 
Graph: Credit Score (300-900) vs Loan Amount (₹)
🟢 Eligible = High CIBIL (>650), 🔴 Not Eligible = Low CIBIL

**Tech Stack:** Python, Pandas, NumPy, Matplotlib, Scikit-learn
**Accuracy:** 0.9075

**How to Run:**
```bash
pip install pandas numpy matplotlib scikit-learn
python eligibility.py
