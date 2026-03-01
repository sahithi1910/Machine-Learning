import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("housing.csv")

# Step 2: Remove target variable (assuming 'median_house_value' is the target)
X = df.drop("median_house_value", axis=1)

# Step 3: Encode categorical variables (like 'ocean_proximity')
X = pd.get_dummies(X, drop_first=True)

# Step 4: Handle missing values
X = X.dropna()

# Step 5: (Optional) Sample data to speed up computation
# Comment this out if you want to use the full dataset
X_sample = X.sample(5000, random_state=42)

# Step 6: Elbow method
distortions = []
k_values = range(2, 20)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_sample)
    distortions.append(kmeans.inertia_)   # inertia = sum of squared distances to nearest cluster center

# Step 7: Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, distortions, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Distortion (Inertia)")
plt.grid(True)
plt.show()
