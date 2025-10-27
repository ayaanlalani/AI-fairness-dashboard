import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv('diabetes_cleaned.csv')

# Define the feature columns and the target variable (loan default)
features = ['Pregnancies', 'older_age', 'high_pregnancy_count']
target = 'Outcome'

# Split data into features and target
X = df[features]
y = df[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine X_train and y_train back into a single DataFrame
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Convert the pandas dataframe into a BinaryLabelDataset (required for AIF360 metrics)
train_dataset = BinaryLabelDataset(df=train_df, label_names=[target], protected_attribute_names=['Pregnancies', 'older_age', 'high_pregnancy_count'])
test_dataset = BinaryLabelDataset(df=test_df, label_names=[target], protected_attribute_names=['Pregnancies', 'older_age', 'high_pregnancy_count'])

# Instantiate the DemographicParity object from AIF360
dp = DemographicParity()
dp_score = dp.compute_metrics(test_dataset)
print(f"Demographic Parity: {dp_score}")

# Instantiate the EqualOpportunity object from AIF360
eo = EqualOpportunity()
eo_score = eo.compute_metrics(test_dataset)
print(f"Equal Opportunity: {eo_score}")

# Instantiate the DisparateImpact object from AIF360
di = DisparateImpact()
di_score = di.compute_metrics(test_dataset)
print(f"Disparate Impact: {di_score}")

# Instantiate the AverageOddsDifference object from AIF360
aod = AverageOddsDifference()
aod_score = aod.compute_metrics(test_dataset)
print(f"Average Odds Difference: {aod_score}")
