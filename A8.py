import pandas as pd
import numpy as np
df = pd.read_excel("Data.xlsx", sheet_name="thyroid0387_UCI")
print("Missing values BEFORE imputation:\n")
print(df.isnull().sum())
numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    if ((df[col] < lower) | (df[col] > upper)).any():
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("\nMissing values AFTER imputation:\n")
print(df.isnull().sum())
