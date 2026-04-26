# =========================================
# A5: LIME + SHAP EXPLAINABILITY
# =========================================

# 1. INSTALL (run once in terminal)
# pip install lime shap

# =========================================
# 2. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import lime
import lime.lime_tabular
import shap

# =========================================
# 3. LOAD DATASET
# =========================================
df = pd.read_csv("Feature_data.csv")

# =========================================
# 4. CLEAN DATA
# =========================================
df.columns = df.columns.str.strip()

# Drop unnecessary columns safely
df = df.drop(["sentence", "para_id"], axis=1, errors='ignore')

# Keep numeric only
df = df.select_dtypes(include=[np.number])

# Handle missing values
df = df.fillna(df.mean())

# =========================================
# 5. SPLIT FEATURES & TARGET
# =========================================
X = df.drop("label", axis=1)
y = df["label"]

# =========================================
# 6. TRAIN-TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 7. SCALING
# =========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================
# 8. TRAIN MODEL
# =========================================
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")

# =========================================
# 🔶 PART 1: LIME EXPLANATION
# =========================================
print("\nGenerating LIME explanation...")

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled,
    feature_names=X.columns.tolist(),
    class_names=[str(c) for c in np.unique(y)],
    mode="classification"
)

# Explain a single test instance
i = 0  # you can change index

exp = explainer_lime.explain_instance(
    X_test_scaled[i],
    model.predict_proba
)

# Show explanation
exp.show_in_notebook()

# Save as HTML file
exp.save_to_file("lime_explanation.html")

print("LIME explanation saved as lime_explanation.html")

# =========================================
# 🔶 PART 2: SHAP EXPLANATION
# =========================================
print("\nGenerating SHAP explanation...")

# Create SHAP explainer
explainer_shap = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer_shap.shap_values(X_test_scaled)

# -----------------------------------------
# SHAP Summary Plot (GLOBAL)
# -----------------------------------------
shap.summary_plot(shap_values, X_test_scaled)

# -----------------------------------------
# SHAP Force Plot (LOCAL)
# -----------------------------------------
shap.initjs()

# For first sample
shap.force_plot(
    explainer_shap.expected_value[0],
    shap_values[0][i],
    X_test_scaled[i],
    feature_names=X.columns.tolist()
)

print("SHAP analysis completed!")
