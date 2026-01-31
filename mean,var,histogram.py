import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
df=pd.read_csv("Feature_data.csv")
feature=df["F1"].dropna()
mean1=np.mean(feature)
var1=np.var(feature)
count,bins=np.histogram(feature,bins=10)
if __name__=="__main__":
    print("Mean of the feature vector",mean1)
    print("Variance of the feature vector",var1)
    print("Histogram counts",count)
    print("Bin Ranges",bins)
    plt.hist(feature,bins=10,edgecolor="black")
    plt.xlabel("Feature_Value")
    plt.ylabel("Frequency")
    plt.show()
