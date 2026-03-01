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

# Step 5: Fit KMeans (start with 2 clusters as in A4)
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X)

# Step 6: Get results
labels = kmeans.labels_
centers = kmeans.cluster_centers_

print("Cluster labels (first 10):", labels[:10])
print("Cluster centers shape:", centers.shape)

# Step 7: Evaluate clustering quality
sil_score = silhouette_score(X, labels)
ch_score = calinski_harabasz_score(X, labels)
db_index = davies_bouldin_score(X, labels)

print("Silhouette Score:", sil_score)
print("Calinski-Harabasz Score:", ch_score)
print("Davies-Bouldin Index:", db_index)

# Step 8: Optional visualization (using first two features only)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centers')
plt.title("KMeans Clustering on Housing Data")
plt.legend()
plt.show()
