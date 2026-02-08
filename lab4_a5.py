import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
np.random.seed(42)
X=np.random.randint(1,11,size=20)
Y=np.random.randint(1,11,size=20)
y_train=[]
for i in range(20):
    y_train.append([X[i],Y[i]])
y_train=np.array(y_train)
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
k_values=[1, 3, 5, 9]
plt.figure(figsize=(14, 10))
for i, k in enumerate(k_values):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions=knn.predict(test_points)
    plt.subplot(2, 2, i + 1)
    plt.scatter(
        test_points[:, 0],
        test_points[:, 1],
        c=predictions,
        cmap='bwr',
        alpha=0.3
    )
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap='bwr',
        edgecolor='black'
    )

    plt.title(f"kNN Classification (k = {k})")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.grid(True)

plt.tight_layout()
plt.show()
