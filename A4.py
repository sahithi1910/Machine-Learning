import numpy as np
import pandas as pd
df=pd.read_excel("Data.xlsx",sheet_name="thyroid0387_UCI")
df.info()
df.dtypes
numeric_cols=df.select_dtypes(include=["int64","float64"]).columns
categorical_cols=df.select_dtypes(include=["object"]).columns
print("Numeric colums",numeric_cols)
print("Catgory cols",categorical_cols)
for col in categorical_cols:
    print("Column",col)
    print("Unique values",df[col].unique())
print("Numeric column ranges")
for col in numeric_cols:
    print(f"{col}:min={df[col].min()} max={df[col].max()}")
data_range=df.select_dtypes(include="number").describe
print(data_range)
df.replace("?",np.nan,inplace=True)
missing_values = df.isnull().sum()
print(missing_values)
Q1 = df["TSH"].quantile(0.25)
Q3 = df["TSH"].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df["TSH"] < Q1 - 1.5*IQR) | (df["TSH"] > Q3 + 1.5*IQR)]
print("Number of TSH outliers:", outliers.shape[0])
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {outliers.shape[0]} outliers")
mean_values = df[numeric_cols].mean()
variance_values = df[numeric_cols].var()
std_values = df[numeric_cols].std()

print("Mean:\n", mean_values)
print("\nVariance:\n", variance_values)
print("\nStandard Deviation:\n", std_values)


