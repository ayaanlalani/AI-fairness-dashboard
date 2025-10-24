import pandas as pd
import numpy as np

X = pd.read_csv('data/german_credit_kaggle.csv')

from ucimlrepo import fetch_ucirepo 
statlog_german_credit_data = fetch_ucirepo(id=144) 
y = statlog_german_credit_data.data.targets 
original_features = statlog_german_credit_data.data.features

# general order of reliability 4(most), 1(least)
credit_history_mapping = {
    'A30': 4,  # no credits taken/ all credits paid back on time (most reliable)
    'A31': 3,  # all credits at this bank paid back on time
    'A32': 2,  # existing credits paid back on time till now
    'A33': 1,  # delay in paying off in the past
    'A34': 2   # complex credit situation, not necessarily unreliable
}

credit_history = original_features.iloc[:, 2]  # Attribute 3 is credit history
credit_history_reliability = credit_history.map(credit_history_mapping)

X['Checking account'] = X['Checking account'].fillna('Unknown')
X['Saving accounts'] = X['Saving accounts'].fillna('Unknown')

if len(X) == len(y) == len(credit_history_reliability):
    X['credit reliability'] = credit_history_reliability
    X['credit risk'] = y.iloc[:, 0]
    
    X.to_csv('data/german_credit_CLEANED_dataset.csv', index=False)
    
else:
    print("Error: Number of rows do not match.")
    print(f"X rows: {len(X)}, y rows: {len(y)}, credit history rows: {len(credit_history_reliability)}")
