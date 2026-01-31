import pandas as pd
import numpy as np
import math
df = pd.read_csv("Feature_data.csv")
feature_cols = [col for col in df.columns if col.startswith("F")]
A = df.loc[0, feature_cols].values
B = df.loc[1, feature_cols].values

print("Vector A:", A)
print("Vector B:", B)
def dot_product_manual(A, B):
    result = 0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result
def euclidean_norm_manual(V):
    sum_sq = 0
    for val in V:
        sum_sq += val ** 2
    return math.sqrt(sum_sq)

dot_manual = dot_product_manual(A, B)
dot_numpy = np.dot(A, B)

norm_A_manual = euclidean_norm_manual(A)
norm_A_numpy = np.linalg.norm(A)

norm_B_manual = euclidean_norm_manual(B)
norm_B_numpy = np.linalg.norm(B)
if __name__=="__main__":
    print("Manual Dot Product:", dot_manual)
    print("NumPy Dot Product :", dot_numpy)
    print("Manual Norm of A :", norm_A_manual)
    print("NumPy Norm of A  :", norm_A_numpy)

    print("Manual Norm of B :", norm_B_manual)
    print("NumPy Norm of B  :", norm_B_numpy)
