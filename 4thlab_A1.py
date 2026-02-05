import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report 
df=pd.read_csv("Feature_data.csv")
feature_cols = [col for col in df.columns if col.startswith("F")]
X = df[feature_cols].values
y = df["label"].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
y_pred=neigh.predict(X_test)
y_train_pred=neigh.predict(X_train)
confusion_matrix1=confusion_matrix(y_test,y_pred)
test_report=classification_report(y_test,y_pred)
train_report=classification_report(y_train,y_train_pred)
if __name__=="__main__":
    print("Testing data Report")
    print(test_report)
    print("Training data Report")
    print(train_report)
