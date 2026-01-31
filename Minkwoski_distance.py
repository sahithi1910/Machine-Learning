import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
df=pd.read_csv("Feature_data.csv")
feature_col=[col for col in df.columns if col.startswith("F")]
A=df.loc[0,feature_col].values
B=df.loc[1,feature_col].values
def Minkwoski_distance(arr1,arr2,p):
    sum1=0
    for i in range(len(A)):
        sum1+=abs(arr1[i]-arr2[i])**p
    return sum1**1/p
p_values = list(range(1, 11))
distances = []

for p in p_values:
    d = Minkwoski_distance(A, B, p)
    distances.append(d)
    print(f"Minkowski distance (p={p}): {d}")

plt.plot(p_values, distances, marker='o')
plt.xlabel("p value")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance vs p")
plt.grid(True)
plt.show()

