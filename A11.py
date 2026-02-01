import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("Feature_data.csv")
X =data.iloc[:, :-1]
y=data.iloc[:, -1]
# Convert continuous target to classes
y =pd.cut(y, bins=2, labels=[0, 1])
# Handle categorical features
X =pd.get_dummies(X)
# Train-test split
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
# NN (k = 1)
nn =KNeighborsClassifier(n_neighbors=1)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
acc_nn=accuracy_score(y_test, y_pred_nn)
print("NN Classifier (k = 1) Accuracy:", acc_nn)

# kNN (k = 3)
knn3=KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)
y_pred_knn3=knn3.predict(X_test)
acc_knn3 = accuracy_score(y_test, y_pred_knn3)
print("kNN Classifier (k = 3) Accuracy:", acc_knn3)

print("\nAccuracy for different k values:\n")

# Vary k from 1 to 11
k_values=range(1, 12)
accuracies=[]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"k = {k}  -->  Accuracy = {acc}")

# Best k
best_k = k_values[accuracies.index(max(accuracies))]
best_acc = max(accuracies)

print("\nBest k value:", best_k)
print("Highest Accuracy:", best_acc)

# Plot
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.xticks(k_values)
plt.grid()
plt.show()
