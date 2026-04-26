# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ================================
# 2. LOAD DATASET
# ================================
# Replace with your file path
df = pd.read_csv("Feature_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ================================
# 3. HANDLE NON-NUMERIC DATA (IMPORTANT)
# ================================
# Keep only numeric columns
df_numeric = df.select_dtypes(include=[np.number])

print("Numeric Data Shape:", df_numeric.shape)

# ================================
# 4. COMPUTE CORRELATION MATRIX
# ================================
corr_matrix = df_numeric.corr()

print("\nCorrelation Matrix Computed")

# ================================
# 5. FULL HEATMAP (FOR SMALL DATA)
# ⚠️ May be cluttered for 768 features
# ================================
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.2)

plt.title("Full Feature Correlation Heatmap")
plt.show()

# ================================
# 6. FIND HIGHLY CORRELATED FEATURES
# ================================
threshold = 0.8  # You can change this (0.7–0.9)

corr_matrix_abs = corr_matrix.abs()

# Upper triangle matrix
upper = corr_matrix_abs.where(
    np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool)
)

# Find features to drop
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print("\nHighly Correlated Features to Drop:")
print(to_drop)
print("Number of features to drop:", len(to_drop))

# ================================
# 7. REDUCE DATASET
# ================================
df_reduced = df_numeric.drop(columns=to_drop)

print("\nReduced Dataset Shape:", df_reduced.shape)

# ================================
# 8. HEATMAP AFTER REMOVAL (CLEAN)
# ================================
plt.figure(figsize=(10,8))
sns.heatmap(df_reduced.corr(), cmap='coolwarm', linewidths=0.2)

plt.title("Heatmap After Removing Highly Correlated Features")
plt.show()

# ================================
# 9. OPTIONAL: CLUSTERED HEATMAP (BEST FOR MANY FEATURES)
# ================================
sns.clustermap(df_reduced.corr(),
               cmap='coolwarm',
               figsize=(10,10))

plt.title("Clustered Correlation Heatmap")
plt.show()

# ================================
# 10. SAVE REDUCED DATASET
# ================================
df_reduced.to_csv("reduced_dataset.csv", index=False)

print("\nReduced dataset saved as 'reduced_dataset.csv'")
