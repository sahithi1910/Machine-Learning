import pandas as pd
import numpy as np
df=pd.read_excel("Data.xlsx", sheet_name="thyroid0387_UCI")
df=df.replace({'t': 1, 'f': 0})
numeric_cols=df.select_dtypes(include=np.number).columns
non_binary_cols=[]
for col in numeric_cols:
    if not set(df[col].dropna().unique()).issubset({0, 1}):
        non_binary_cols.append(col)
normalized_df=df.copy()
for col in non_binary_cols:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3 - Q1
    lower=Q1 - 1.5 * IQR
    upper=Q3 + 1.5 * IQR
    if ((df[col]<lower)|(df[col] > upper)).any():
        mean=df[col].mean()
        std=df[col].std()
        normalized_df[col]=(df[col] - mean) / std
    else:
        min_val=df[col].min()
        max_val=df[col].max()
        normalized_df[col]=(df[col] - min_val) / (max_val - min_val)
print("Original Data (first 5 rows):\n")
print(df.head())
print("\nNormalized Data (first 5 rows):\n")
print(normalized_df.head())
