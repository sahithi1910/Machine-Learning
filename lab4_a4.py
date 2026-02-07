import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
np.random.seed(42)
X=np.random.randint(1,11,size=20)
Y=np.random.randint(1,11,size=20)
labels=[]
for i in range(20):
    labels.append([X[i],Y[i]])
labels=np.array(labels)
y_train=[]
for i in range(20):
    if X[i]+Y[i]>10:
       y_train.append(1)
    else:
        y_train.append(0)
y_train=np.array(y_train)
for i in range(20):
    if y_train[i]==0:
        plt.scatter(X[i],Y[i],color='blue')
    else:
        plt.scatter(X[i],Y[i],color='red')
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("Training Data")
plt.grid(True)
plt.show()
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(labels, y_train)
x_test=np.arange(0, 10, 0.1)
y_test=np.arange(0, 10, 0.1)

XX, YY=np.meshgrid(x_test, y_test)
test_data=np.c_[XX.ravel(), YY.ravel()]
test_predictions=knn.predict(test_data)
plt.figure(figsize=(8, 6))

for i in range(len(test_data)):
    if test_predictions[i]==0:
        plt.scatter(test_data[i, 0], test_data[i, 1], color='blue', s=5)
    else:
        plt.scatter(test_data[i, 0], test_data[i, 1], color='red', s=5)
for i in range(20):
    plt.scatter(X_train[i], Y_train[i], color='black', edgecolors='white', s=100)
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("kNN Classification (k=3) – Decision Boundary")
plt.grid(True)
plt.show()

