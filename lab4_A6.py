import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_excel("Data.xlsx", sheet_name="marketing_campaign")

data = df[['Income', 'MntWines', 'Response']]
data = data.fillna(data.mean())

X = data[['Income', 'MntWines']].values
y = data['Response'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1])
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1])
plt.title("Training Data")
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
colors = ['blue' if p == 0 else 'red' for p in pred]

plt.scatter(X_test[:, 0], X_test[:, 1], c=colors)
plt.title("Test Data")
plt.show()
