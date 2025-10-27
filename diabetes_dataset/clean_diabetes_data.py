import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/diabetes.csv')

df_clean = df.copy()
df_clean = df_clean.dropna(subset=['Pregnancies', 'Age', 'Outcome'])

invalid_zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
df_clean = df_clean[(df_clean[invalid_zero_features] != 0).all(axis=1)]
df_clean = df_clean[(df_clean['Age'] > 0) & (df_clean['Age'] < 110)]
df_clean = df_clean[df_clean['Pregnancies'] >= 0]

# Encode categorical variables
# encoder = LabelEncoder()
# df['gender'] = encoder.fit_transform(df['gender'])  # e.g., Male = 1, Female = 0

df_clean['older_age'] = (df_clean['Age'] >= 35).astype(int)
df_clean['high_pregnancy_count'] = (df_clean['Pregnancies'] >= 3).astype(int)

df_clean.to_csv('data/diabetes_cleaned.csv', index=False)