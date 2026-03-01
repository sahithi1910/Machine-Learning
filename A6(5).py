import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("housing.csv")

# Step 2: Remove target variable (assuming 'median_house_value' is the target)
X = df.drop("median_house_value", axis=1)

# Step 3: Encode categorical variables (like 'ocean_proximity')
X = pd.get_dummies(X, drop_first=True)

# Step 4: Handle missing values
X = X.dropna()

# Step 5: Initialize lists to store scores
silhouette_scores = []
ch_scores = []
db_scores = []
k_values = range(2, 11)  # test k from 2 to 10

# Step 6: Loop over different k values
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    labels = kmeans.labels_
    
    # Compute metrics
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    
    silhouette_scores.append(sil)
    ch_scores.append(ch)
    db_scores.append(db)
    
    print(f"k={k}: Silhouette={sil:.4f}, CH={ch:.4f}, DB={db:.4f}")

# Step 7: Plot metrics vs k
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")

plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o')
plt.title("Calinski-Harabasz Score vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("CH Score")

plt.subplot(1, 3, 3)
plt.plot(k_values, db_scores, marker='o')
plt.title("Davies-Bouldin Index vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("DB Index")

plt.tight_layout()
plt.show()
