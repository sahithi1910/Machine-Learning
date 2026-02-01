import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
data = pd.read_csv("Feature_data.csv")

# Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert continuous target to classes
y = pd.cut(y, bins=2, labels=[0, 1])

# Encode categorical features
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
if __name__ == "__main__":
    print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Classification Report:\n", classification_report(y_train, y_train_pred))

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
