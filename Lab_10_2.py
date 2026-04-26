# =========================================
# A2: PCA WITH 99% EXPLAINED VARIANCE
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

# 5. DEFINE TARGET
target_column = "label"
# Drop unnecessary columns
df = df.drop(["sentence", "para_id"], axis=1, errors='ignore')

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

# 8. APPLY PCA (99%)
pca_99 = PCA(n_components=0.99)

X_train_pca = pca_99.fit_transform(X_train_scaled)
X_test_pca = pca_99.transform(X_test_scaled)

print("Original features:", X.shape[1])
print("Reduced features (99%):", X_train_pca.shape[1])

# 9. TRAIN MODEL
model = RandomForestClassifier(random_state=42)
model.fit(X_train_pca, y_train)

# 10. PREDICT
y_pred = model.predict(X_test_pca)

# 11. EVALUATE
accuracy = accuracy_score(y_test, y_pred)

print("\nA2 RESULTS (99% VARIANCE)")
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# =========================================
# 12. SAVE RESULTS TO EXCEL
# =========================================

# 👉 Give your Excel file name here
file_name = "text_results.xlsx"

# Convert classification report to dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Create accuracy DataFrame
accuracy_df = pd.DataFrame({
    "Metric": ["Accuracy"],
    "Value": [accuracy]
})

# Write to Excel
with pd.ExcelWriter(file_name) as writer:
    accuracy_df.to_excel(writer, sheet_name="Accuracy", index=False)
    report_df.to_excel(writer, sheet_name="Classification_Report")

print(f"\n✅ Results saved successfully in '{file_name}'")
