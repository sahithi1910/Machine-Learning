import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("Feature_data.csv")
df = df[df["label"].isin([0, 1])]
feature_cols = [col for col in df.columns if col.startswith("F")]

X = df[feature_cols].values
y = df["label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
sk_accuracy = knn.score(X_test, y_test)
print("\nSklearn kNN Accuracy:", sk_accuracy)
sk_predictions = knn.predict(X_test)
print("First 5 Predictions (Sklearn):", sk_predictions[:5])

# Predicting one test vector
single_prediction = knn.predict([X_test[0]])
print("Prediction for first test vector:", single_prediction[0])

# Euclidean distance function
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Own kNN function
def my_knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            dist = euclidean_distance(test_point, X_train[i])
            distances.append((dist, y_train[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        labels = [label for _, label in k_nearest]
        predicted_label = Counter(labels).most_common(1)[0][0]

        predictions.append(predicted_label)

    return np.array(predictions)

# Accuracy function
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Testing own kNN
my_predictions = my_knn_predict(X_train, y_train, X_test, k=3)
my_accuracy = accuracy(y_test, my_predictions)

print("\nMy kNN Accuracy:", my_accuracy)
print("First 5 Predictions (My kNN):", my_predictions[:5])
print("\nAccuracy Comparison:")
print("Sklearn kNN:", sk_accuracy)
print("My kNN     :", my_accuracy)
