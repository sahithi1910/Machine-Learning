import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
file_path="Data.xlsx"
df=pd.read_excel(file_path, sheet_name="marketing_campaign")
data=df[['Income', 'MntWines', 'Response']]
data=data.fillna(data.mean())
X =data[['Income', 'MntWines']]
y =data['Response']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

params={"n_neighbors": [1, 3, 5, 7, 9, 11, 13]}
grid=GridSearchCV(KNeighborsClassifier(), params, cv=5)
grid.fit(X_train, y_train)

best=grid.best_estimator_
pred=best.predict(X_test)

print("Best k:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
