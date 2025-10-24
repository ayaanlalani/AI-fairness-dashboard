import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/german_credit_CLEANED_dataset.csv')

features = [
    'Age', 
    'Sex', 
    'Job', 
    'Housing', 
    'Saving accounts', 
    'Checking account', 
    'Credit amount', 
    'Duration', 
    'Purpose', 
    'credit reliability'
]
target = 'credit risk'

categorical_features = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
for col in categorical_features:
    df[col] = pd.Categorical(df[col]).codes

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = ['Age', 'Credit amount', 'Duration', 'credit reliability']
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

def prepare_binary_label_dataset(X, y, y_pred_prob, features, label_name, protected_attribute_name):
    df_temp = X.copy()
    df_temp[label_name] = y
    
    return BinaryLabelDataset(
        df=df_temp,
        label_names=[label_name],
        protected_attribute_names=[protected_attribute_name],
        favorable_label=1,  
        unfavorable_label=2 
    )

# Prepare datasets
test_dataset = prepare_binary_label_dataset(
    X_test, 
    y_test, 
    y_pred_prob,
    features, 
    target, 
    'Sex'
)

# Predicted dataset
pred_dataset = prepare_binary_label_dataset(
    X_test, 
    y_pred, 
    y_pred_prob,
    features, 
    target, 
    'Sex'
)

# Compute fairness metrics
def compute_fairness_metrics(test_dataset, pred_dataset):
    unprivileged_groups = [{'Sex': 0}]
    privileged_groups = [{'Sex': 1}]   
    
    dataset_metric = BinaryLabelDatasetMetric(
        test_dataset, 
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    classification_metric = ClassificationMetric(
        test_dataset, 
        pred_dataset,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    demographic_parity = dataset_metric.mean_difference()
    disparate_impact = dataset_metric.disparate_impact() 
    equal_opportunity = classification_metric.equal_opportunity_difference()
    average_odds_difference = classification_metric.average_odds_difference()

    fairness_metrics = {
        'DP': round(demographic_parity, 3),
        'DI': round(disparate_impact, 3),
        'EO': round(equal_opportunity, 3),
        'AOD': round(average_odds_difference, 3),
    }
    
    return fairness_metrics


fairness_results = compute_fairness_metrics(test_dataset, pred_dataset)
print("\n\n=====================================================================")
print("Target=Credit Risk, Undeprivileged: Sex(Female), Privilege: Sex(Male)")
print("DP: ", fairness_results['DP'])
print("DI: ", fairness_results['DI'])
print("EO: ", fairness_results['EO'])
print("AOD: ", fairness_results['AOD'])