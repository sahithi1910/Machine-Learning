import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
data=pd.read_csv("Feature_data.csv")
X=pd.get_dummies(data.iloc[:, :-1])
y=data.iloc[:, -1].values
y=np.where(y >= np.mean(y), 1, 0)
X_train, X_test, y_train, y_test=train_test_split(
    X.values, y, test_size=0.3, random_state=42
)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_knn)
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

theta = np.linalg.pinv(X_train_b) @ y_train

y_pred_linear = np.where(X_test_b @ theta >= 0.5, 1, 0)

acc_linear = accuracy_score(y_test, y_pred_linear)

print("kNN Classifier Accuracy:", acc_knn)
print("Matrix Inversion Technique Accuracy:", acc_linear)
