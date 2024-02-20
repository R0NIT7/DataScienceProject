import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  r2_score
from sklearn.linear_model import LinearRegression, Lasso

f=pd.read_csv(r"C:\Users\sunan\Desktop\archive\p\tvmarketing.csv")
df=pd.DataFrame(f)


X=df.drop(['Sales'],axis=1)
y=df['Sales']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.33,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
print("intercept: ",model.intercept_)
print("test accuracy is ",r2_score(y_test,y_predict))

lasso=Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
y_pred_train_lasso=lasso.predict(X_train)
y_pred_test_lasso=lasso.predict(X_test)
print("R2 training accuracy: ",r2_score(y_train,y_pred_train_lasso))
print("R2 testing accuracy: ",r2_score(y_test,y_pred_test_lasso))

