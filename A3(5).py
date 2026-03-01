import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("housing.csv")
print(df.head())
print(df.columns)
print(df.isnull().sum())  # check for missing values
df = df.dropna()
df = pd.get_dummies(df, columns=["ocean_proximity"])
X = df.drop("median_house_value", axis=1)  # all columns except target
Y = df["median_house_value"]
print("Features used:", X.columns.tolist())
print("Shape of X:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
def get_metrics(y_actual, y_predicted):
    mse = mean_squared_error(y_actual, y_predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100
    r2 = r2_score(y_actual, y_predicted)
    return mse, rmse, mape, r2

mse_train, rmse_train, mape_train, r2_train = get_metrics(y_train, pred_train)
mse_test, rmse_test, mape_test, r2_test = get_metrics(y_test, pred_test)
print("Mean Squared Error of train :",mse_train)
print("Root Mean Squared Error of train :",rmse_train)
print("Mean Absolute Percentage Error of train:",mape_train)
print("R2 Score of train:",r2_train)
print("Mean Squared Error of train :",mse_test)
print("Root Mean Squared Error of test :",rmse_test)
print("Mean Absolute Percentage Error of test:",mape_test)
print("R2 Score of test:",r2_test)
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\nCoefficients:")
print(coef_df)
