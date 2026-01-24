import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm
df = pd.read_excel("Data.xlsx",sheet_name="thyroid0387_UCI")
df = df.replace({'t':1, 'f':0})
df20 = df.iloc[:20]
binary_cols = []
for col in df20.columns:
    unique_vals = df20[col].dropna().unique()
    if set(unique_vals).issubset({0, 1}):
        binary_cols.append(col)
binary_data=df20[binary_cols].to_numpy()
numeric_data=df20.select_dtypes(include='number').to_numpy()
JC=np.zeros((20, 20))
SMC=np.zeros((20, 20))
COS=np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        v1 =binary_data[i]
        v2=binary_data[j]
        f11=np.sum((v1==1)&(v2==1))
        f00=np.sum((v1==0)&(v2==0))
        f10=np.sum((v1==1)&(v2==0))
        f01=np.sum((v1==0)&(v2==1))
        JC[i, j]=f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
        SMC[i, j]=(f11 + f00) / (f11 + f10 + f01 + f00)
        a=numeric_data[i]
        b=numeric_data[j]
        COS[i, j] = np.dot(a, b) / (norm(a) * norm(b))
print("JC:",JC)
print("SMC:",SMC)
print("COS:",COS)
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
sns.heatmap(JC, annot=True, cmap="Blues")
plt.title("Jaccard Coefficient Heatmap")
plt.subplot(1, 3, 2)
sns.heatmap(SMC, annot=True, cmap="Greens")
plt.title("Simple Matching Coefficient Heatmap")
plt.subplot(1, 3, 3)
sns.heatmap(COS, annot=True, cmap="Reds")
plt.title("Cosine Similarity Heatmap")
plt.tight_layout()
plt.show()
