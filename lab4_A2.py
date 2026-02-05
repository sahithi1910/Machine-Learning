import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
df=pd.read_excel("Data.xlsx",sheet_name="IRCTC Stock Price" )
neigh=KNeighborsRegressor(n_neighbors=3)
X=df[["Price","Open","High","Low"]]
y=df["Chg%"].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
neigh.fit(X_train,y_train)
y_pred=neigh.predict(X_test)
MSE=mean_squared_error(y_test,y_pred)
RMSE=np.sqrt(MSE)
mape = mean_absolute_percentage_error(y_test, y_pred)
r_score=r2_score(y_test, y_pred)
if __name__=="__main__":
    print("MSE:",MSE)
    print("RMSE:",RMSE)
    print("MAPE:",mape)
    print("R2 score",r_score)
