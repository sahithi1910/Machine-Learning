import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Feature_data.csv")

X=data.iloc[:, :-1]
y=data.iloc[:, -1]
y=pd.cut(y, bins=2, labels=[0, 1])

X=pd.get_dummies(X)

X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

def evaluate(y_true, y_pred, beta=1):
    TP = TN = FP = FN = 0

    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            TP += 1
        elif t == 0 and p == 0:
            TN += 1
        elif t == 0 and p == 1:
            FP += 1
        elif t == 1 and p == 0:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f_beta = ((1 + beta**2) * precision * recall) / (
        (beta**2 * precision) + recall
    ) if (precision + recall) else 0

    return TP, TN, FP, FN, accuracy, precision, recall, f_beta

TP, TN, FP, FN, acc, prec, rec, f1 = evaluate(y_train, y_train_pred)
print("\nTRAIN DATA")
print("Confusion Matrix:", [[TN, FP], [FN, TP]])
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
TP, TN, FP, FN, acc, prec, rec, f1 = evaluate(y_test, y_test_pred)
print("\nTEST DATA")
print("Confusion Matrix:", [[TN, FP], [FN, TP]])
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
