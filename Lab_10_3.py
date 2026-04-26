# =========================================
# A3: PCA WITH 95% EXPLAINED VARIANCE
# =========================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. LOAD DATASET
df = pd.read_csv("Feature_data.csv")

# 3. KEEP NUMERIC DATA
df = df.select_dtypes(include=[np.number])

# 4. HANDLE MISSING VALUES
df = df.fillna(df.mean())

target_column = "label"
# Drop unnecessary columns
df = df.drop(["sentence", "para_id"], axis=1, errors='ignore')
5
# Define X and y
X = df.drop("label", axis=1)
y = df["label"]


# 6. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. APPLY PCA (95%)
pca_95 = PCA(n_components=0.95)

X_train_pca = pca_95.fit_transform(X_train_scaled)
X_test_pca = pca_95.transform(X_test_scaled)

print("Original features:", X.shape[1])
print("Reduced features (95%):", X_train_pca.shape[1])

# 9. TRAIN MODEL
model = RandomForestClassifier(random_state=42)
model.fit(X_train_pca, y_train)

# 10. PREDICT
y_pred = model.predict(X_test_pca)

# 11. EVALUATE
accuracy = accuracy_score(y_test, y_pred)

print("\nA3 RESULTS (95% VARIANCE)")
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
