import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
X=np.random.randint(1,11,size=20)
Y=np.random.randint(1,11,size=20)
labels=[]
for i in range(20):
    if X[i]+Y[i]>10:
        labels.append(1)
    else:
        labels.append(0)
for i in range(20):
    if labels[i]==0:
        plt.scatter(X[i],Y[i],color='blue')
    else:
        plt.scatter(X[i],Y[i],color='red')
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("Training Data ")
plt.grid(True)
plt.show()
