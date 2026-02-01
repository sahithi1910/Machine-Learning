import pandas as pd
import numpy as np
import math
df = pd.read_csv("Feature_data.csv")

numeric_df = df.select_dtypes(include='number')

def mean(data):
    return np.mean(data)

def std_dev(data):
    return np.std(data)


for col in numeric_df.columns:
    data = numeric_df[col].values   
    print(f"\nFeature: {col}")
    print("Mean:", mean(data))
    print("Std Dev:", std_dev(data))

labels = df.iloc[:, -1]

unique_classes = labels.unique()

class1 = unique_classes[0]
class2 = unique_classes[1]

data1 = numeric_df[labels == class1]
data2 = numeric_df[labels == class2]

centroid1 = np.mean(data1, axis=0)
centroid2 = np.mean(data2, axis=0)

print("Centroid of Class", class1)
print(centroid1)

print("Centroid of Class", class2)
print(centroid2)

spread1 = np.std(data1, axis=0)
spread2 = np.std(data2, axis=0)

print("Intraclass Spread of Class", class1)
print(spread1)

print("Intraclass Spread of Class", class2)
print(spread2)


interclass_distance = np.linalg.norm(centroid1 - centroid2)

print("Interclass Distance between two classes:")
print(interclass_distance)
