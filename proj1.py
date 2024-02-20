
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree



f=pd.read_csv(r"C:\Users\sunan\Desktop\archive\p\suv_data.csv")
df=pd.DataFrame(f)
df.drop(['User ID'],inplace=True, axis=1)
df['Gender']=df['Gender'].map({'Female':1,'Male':0})


X=df.drop(['Purchased'],axis=1)
y=df['Purchased']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=1)

model = LogisticRegression(max_iter=4000)
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
confusion_matrix1=confusion_matrix(y_test,y_predict)
print(confusion_matrix1)
print(classification_report(y_test,y_predict.round()))
accuracy=accuracy_score(y_test,y_predict)
accuracy1=str(accuracy*100)
print("Logistic Regression accuracy score is " + accuracy1 +"%")

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
confusion_matrix2=confusion_matrix(y_test,y_pred)
print(confusion_matrix2)
print(classification_report(y_test,y_pred))
accuracy2=accuracy_score(y_test,y_pred)
accuracy2=str(accuracy2*100)
print("Initial accuracy score is " + accuracy2 +"%")

leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))

p=[1,2]
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
knn_2 = KNeighborsClassifier()
clf = GridSearchCV(knn_2, hyperparameters, cv=10)
best_model = clf.fit(X,y)
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
knn2=KNeighborsClassifier(n_neighbors=1, p=1,leaf_size=1)
knn2.fit(X_train,y_train)
y_pred2=knn2.predict(X_test)
confusion_matrix3=confusion_matrix(y_test,y_pred2)
print(confusion_matrix3)
print(classification_report(y_test,y_pred2))
accuracy3=accuracy_score(y_test,y_pred2)
accuracy3=str(accuracy3*100)
print("After hypertuning the model, accuracy score is " + accuracy3 +"%")

new_data=pd.DataFrame({
    'Gender':[0,1,1],
    'Age':[23,47,24],
    'EstimatedSalary':[22000,65000,500000]
})

print("\nPredicted decision for new data:",knn2.predict(new_data))


dt = DecisionTreeClassifier(criterion = 'gini', random_state = 0, max_depth=5 )
dt.fit(X_train, y_train)
y_predt=dt.predict(X_test)
yt_predt =dt.predict(X_train)
confusion_matrix4=confusion_matrix(y_test,y_predt)
print(confusion_matrix4)

print('The Training Accuracy of the algorithm is ', accuracy_score(y_train, yt_predt))
print('The Testing Accuracy of the algorithm is ', accuracy_score(y_test, y_predt))
print(classification_report(y_test,y_predt))

plt.figure(figsize=(50,30))
tree.plot_tree(dt,filled=True,rounded=True,fontsize=10)
plt.show()



score = {}
def eval_model(model, y_true,y_preds):
    prec = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    acc = accuracy_score(y_true, y_preds)

    score[model] = [prec, recall, f1, acc]

eval_model("Log Regression",y_test,y_predict)
eval_model("KNN",y_test,y_pred)
eval_model("KNN2",y_test,y_pred2)
eval_model("Decision Tree",y_test,y_predt)
performance = pd.DataFrame(score,index=["Precision","Recall","F1","Accuracy"])
print(performance)