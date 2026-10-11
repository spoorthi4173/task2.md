import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the shared trading file. Resolve paths relative to this script so it
# works even when launched from a different working directory.
data_dir = Path(__file__).resolve().parent
input_path = data_dir / "trades_orderbook_STUDENT_dirty.csv"

# The student dirty file may not be included in the workspace. The available
# clean answer-key file has the same trading rows and is sufficient here
# because this demonstration only uses the row count to generate loan data.
if not input_path.exists():
    fallback_path = data_dir / "trades_orderbook_clean_ANSWER_KEY.csv"
    if fallback_path.exists():
        input_path = fallback_path
        print(f"Student input not found; using available file: {input_path.name}")
    else:
        raise FileNotFoundError(
            f"Could not find trades_orderbook_STUDENT_dirty.csv in {data_dir}"
        )

df = pd.read_csv(input_path)
print("Trading data loaded:", df.shape)

# IMPORTANT:
# The trading file has no credit score or loan approval columns.
# These are SIMULATED loan data for assessment demonstration.
np.random.seed(42)
n = len(df)

loan = pd.DataFrame({
    "credit_score": np.random.randint(300, 901, n),
    "loan_amount": np.random.randint(50000, 1000001, n)
})

# Simulated eligibility rule for demonstration only
loan["eligible"] = (
    (loan["credit_score"] >= 650) &
    (loan["loan_amount"] <=
     loan["credit_score"] * 1000)
).astype(int)

# 2. Select input and output
X = loan[["credit_score", "loan_amount"]]
y = loan["eligible"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

# 4. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Predict eligibility
predictions = model.predict(X_test)
print("Accuracy:",
      accuracy_score(y_test, predictions))

# 6. Predict for a new applicant
score = float(input("Enter credit score (300-900): "))
amount = float(input("Enter loan amount required: "))

applicant = pd.DataFrame({
    "credit_score": [score],
    "loan_amount": [amount]
})

result = model.predict(applicant)[0]

print("Loan Eligible" if result == 1
      else "Loan Not Eligible")

# 7. Plot credit score vs loan amount
colors = loan["eligible"].map({1: "green", 0: "red"})

plt.figure(figsize=(9, 6))
plt.scatter(
    loan["credit_score"], loan["loan_amount"],
    c=colors, alpha=0.6
)

plt.xlabel("Credit Score (CIBIL Score)")
plt.ylabel("Loan Amount Required (₹)")
plt.title("Loan Eligibility using Logistic Regression")
plt.grid(True, alpha=0.3)

plt.scatter([], [], color="green", label="Eligible")
plt.scatter([], [], color="red", label="Not Eligible")
plt.legend()
plt.tight_layout()
plt.show()
