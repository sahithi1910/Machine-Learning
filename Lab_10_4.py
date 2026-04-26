# =========================================
# A4: SEQUENTIAL FEATURE SELECTION (OPTIMIZED)
# =========================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, classification_report

# =========================================
# 2. LOAD DATASET
# =========================================
df = pd.read_csv("Feature_data.csv")

# =========================================
# 3. CLEAN DATA
# =========================================
df.columns = df.columns.str.strip()

# Drop unnecessary columns safely
df = df.drop(["sentence", "para_id"], axis=1, errors='ignore')

# Keep only numeric
df = df.select_dtypes(include=[np.number])

# Handle missing values
df = df.fillna(df.mean())

# =========================================
# 4. SPLIT FEATURES & TARGET
# =========================================
X = df.drop("label", axis=1)
y = df["label"]

print("Original Features:", X.shape[1])

# =========================================
# 5. TRAIN-TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 6. SCALING
# =========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================
# 🔶 STEP 1: PCA (PRE-REDUCTION)
# =========================================
# Reduce 768 → manageable size first
pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("After PCA Features:", X_train_pca.shape[1])

# =========================================
# 🔶 STEP 2: SEQUENTIAL FEATURE SELECTION
# =========================================
# Use lightweight model for SFS
base_model = LogisticRegression(max_iter=1000)

sfs = SequentialFeatureSelector(
    base_model,
    n_features_to_select=20,   # safe number
    direction="forward",
    scoring="accuracy",
    cv=2,                      # reduced CV
    n_jobs=1                   # IMPORTANT: avoid crash
)

sfs.fit(X_train_pca, y_train)

# Transform data
X_train_sfs = sfs.transform(X_train_pca)
X_test_sfs = sfs.transform(X_test_pca)

print("Final Selected Features:", X_train_sfs.shape[1])

# =========================================
# 🔶 STEP 3: FINAL MODEL TRAINING
# =========================================
from sklearn.ensemble import RandomForestClassifier

final_model = RandomForestClassifier(random_state=42)
final_model.fit(X_train_sfs, y_train)

# Predict
y_pred = final_model.predict(X_test_sfs)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("\nA4 RESULTS (SFS + PCA)")
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================================
# 🔶 STEP 4: SAVE RESULTS TO EXCEL
# =========================================
file_name = "A4_SFS_results.xlsx"

# Convert report
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Accuracy + feature info
summary_df = pd.DataFrame({
    "Metric": ["Accuracy", "Original Features", "After PCA", "After SFS"],
    "Value": [accuracy, X.shape[1], X_train_pca.shape[1], X_train_sfs.shape[1]]
})

with pd.ExcelWriter(file_name) as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    report_df.to_excel(writer, sheet_name="Classification_Report")

print(f"\n✅ Results saved to {file_name}")
