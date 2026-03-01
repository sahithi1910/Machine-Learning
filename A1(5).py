import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv("housing.csv")
X_col=df[["median_income"]]*10000 
Y_col=df["median_house_value"]  

X_train,x_test,y_train,y_test=train_test_split(X_col,Y_col,test_size=0.2,random_state=42)

reg=LinearRegression().fit(X_train,y_train)  
predicted_train=reg.predict(X_train) 

print(predicted_train[:5])
print(y_train[:5]==predicted_train[:5]) 
print(reg.coef_)
print(reg.intercept_) 
