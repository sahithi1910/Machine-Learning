import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler # for scaling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score, recall_score, f1_score,classification_report
import matplotlib.pyplot as plt
def assign_test_class(k):
    np.random.seed(42) # to give same random values every time we run
    X=np.arange(0,10,0.1)  
    Y=np.arange(0,10,0.1)
    xx,yy=np.meshgrid(X,Y) # for covering the entire 2D space
    X_train,Y_train,train_labels=assign_train_class()
    train_points = [[x, y] for x, y in zip(X_train, Y_train)]
    test_points  = [[x, y] for x, y in zip(xx.ravel(), yy.ravel())] # ravel flattens to 1D because scatter can't take 2D
    neigh=KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_points,train_labels)
    test_labels=neigh.predict(test_points)
    plt.figure(figsize=(6,6))
    colors=['blue' if label==0 else 'red' for label in test_labels]
    plt.scatter(
        xx.ravel(),
        yy.ravel(),
        c=colors    )
    plt.xlabel('X Feature')
    plt.ylabel('Y Feature')
    plt.grid(True)
    plt.show()
def different_k():
    for k in [1,5,7]:
        assign_test_class(k)
print(different_k)
