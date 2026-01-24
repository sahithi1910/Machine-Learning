import numpy as np
import pandas as pd
from numpy.linalg import norm
df=pd.read_excel("Data.xlsx",sheet_name="thyroid0387_UCI")
df=df.replace({'t':1,'f':0})
df_numeric= df.select_dtypes(include='number')
v1=df_numeric.iloc[0].to_numpy()
v2=df_numeric.iloc[1].to_numpy()
cosine_similarity=np.dot(v1,v2)/(norm(v1)*norm(v2))
print("Cosine Similarity:",cosine_similarity)
