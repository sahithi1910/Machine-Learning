import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank
df=pd.read_excel("Data.xlsx",sheet_name="Purchase data")
X=df[["Candies (#)","Mangoes (Kg)","Milk Packets (#)"]].to_numpy()
Y=df[["Payment (Rs)"]].to_numpy()
A=df[["Candies (#)","Mangoes (Kg)","Milk Packets (#)","Payment (Rs)"]].to_numpy()
rank=matrix_rank(A)
X_inv= np.linalg.pinv(X)
c=X_inv@Y
print("Rank of the matrix:",rank)
print("Cost of Candies:",c[0][0])
print("Cost of Mangoes:",c[1][0])
print("COst of the milk packets:",c[2][0])

                
                
