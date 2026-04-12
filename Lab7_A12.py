import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Feature_dataset.csv")
df = df.drop(columns=["Subject"])

X = df.drop(columns=["label"]).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print("===== TRAIN METRICS =====")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred, average='weighted'))
print("Recall:", recall_score(y_train, y_train_pred, average='weighted'))
print("F1 Score:", f1_score(y_train, y_train_pred, average='weighted'))

print("\n===== TEST METRICS =====")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_test_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_test_pred, average='weighted'))

print("\n===== PER-LABEL METRICS (TEST) =====")
print(classification_report(y_test, y_test_pred))

labels = np.unique(y)

for label in labels:
    precision = precision_score(y_test, y_test_pred, labels=[label], average='macro')
    recall = recall_score(y_test, y_test_pred, labels=[label], average='macro')
    f1 = f1_score(y_test, y_test_pred, labels=[label], average='macro')

    print(f"\nLabel {label} Metrics:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
