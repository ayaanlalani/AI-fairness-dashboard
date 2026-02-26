"""
clean-germanCreditData.py

Clean and prepare the German Credit dataset for fairness analysis.
Merges the Kaggle dataset with UCI Statlog data to extract:
  - credit_reliability  (ordinal, from credit history codes)
  - foreign_worker       (binary, from UCI Attribute 20 — national origin proxy)
  - AgeGroup             (binary, under_40 / 40+)
  - credit risk          (target variable, 1=good 2=bad)

Protected attributes kept for fairness analysis:
  Sex, AgeGroup, foreign_worker

Usage:
  cd german_credit_dataset
  python clean-germanCreditData.py
"""

import pandas as pd
import numpy as np

X = pd.read_csv("data/german_credit_kaggle.csv")

from ucimlrepo import fetch_ucirepo

statlog_german_credit_data = fetch_ucirepo(id=144)
y = statlog_german_credit_data.data.targets
original_features = statlog_german_credit_data.data.features

# --- Credit history reliability (Attribute 3, column index 2) ---
credit_history_mapping = {
    "A30": 4,  # no credits taken / all paid back on time (most reliable)
    "A31": 3,  # all credits at this bank paid back on time
    "A32": 2,  # existing credits paid back on time till now
    "A33": 1,  # delay in paying off in the past
    "A34": 2,  # complex credit situation, not necessarily unreliable
}
credit_history = original_features.iloc[:, 2]
credit_history_reliability = credit_history.map(credit_history_mapping)

# --- Foreign worker (Attribute 20, column index 19) ---
# A201 = yes (is a foreign worker), A202 = no (not a foreign worker)
foreign_worker_raw = original_features.iloc[:, 19]
foreign_worker = foreign_worker_raw.map({"A201": 1, "A202": 0})

# --- Fill missing categorical values ---
X["Checking account"] = X["Checking account"].fillna("Unknown")
X["Saving accounts"] = X["Saving accounts"].fillna("Unknown")

# --- Derive AgeGroup (40+ aligns with US ECOA age-discrimination threshold) ---
X["AgeGroup"] = np.where(X["Age"] >= 40, "40_plus", "under_40")

# --- Merge UCI-derived columns and save ---
if len(X) == len(y) == len(credit_history_reliability) == len(foreign_worker):
    X["credit reliability"] = credit_history_reliability
    X["foreign_worker"] = foreign_worker.astype(int)
    X["credit risk"] = y.iloc[:, 0]

    X.to_csv("data/german_credit_CLEANED_dataset.csv", index=False)

    print(f"Saved cleaned dataset: {len(X)} rows")
    print(f"  Sex distribution:            {X['Sex'].value_counts().to_dict()}")
    print(f"  AgeGroup distribution:       {X['AgeGroup'].value_counts().to_dict()}")
    print(f"  Foreign worker distribution: {X['foreign_worker'].value_counts().to_dict()}")
    print(f"  Credit risk distribution:    {X['credit risk'].value_counts().to_dict()}")
else:
    print("Error: Row count mismatch across data sources.")
    print(
        f"  Kaggle rows: {len(X)}, UCI targets: {len(y)}, "
        f"credit_history: {len(credit_history_reliability)}, "
        f"foreign_worker: {len(foreign_worker)}"
    )
