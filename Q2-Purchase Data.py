import pandas as pd
import numpy as np
df=pd.read_excel("Data.xlsx",sheet_name="Purchase data")
X=df[["Candies (#)","Mangoes (Kg)","Milk Packets (#)"]].to_numpy()
Y=df[["Payment (Rs)"]].to_numpy()
A=df[["Candies (#)","Mangoes (Kg)","Milk Packets (#)","Payment (Rs)"]].to_numpy()
df["Status"]=np.where(df["Payment (Rs)"]>200,
                    "Rich",
                    "Poor")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
print("Modified excel sheet is given as the output")
df.to_excel("Purchase_data_modified.xlsx",index=False)
                
                
