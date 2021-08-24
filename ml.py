import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.svm import SVC

svc=SVC(kernel='linear')
svc2=SVC(kernel='rbf')
gnb=GaussianNB()
gnb2=BernoulliNB()
gnb3=MultinomialNB()
logr=LogisticRegression(random_state=0)
knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski')
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
mtf = MultiOutputClassifier(rfc, n_jobs=-1)

def train_test(df):
    X=df.iloc[:,0:9]
    Y=df.iloc[:,-1]
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)
    return x_train,x_test,y_train,y_test

def Algorithms(df):
    if len(df['target'].unique())==2:
        RandomForest(df)
        KNN(df)
        LogisticRegression(df)
        NaiveBayes(df)
        SVC(df)
    else:
        RandomForest_multi(df)
def RandomForest(df):
    title='Random Forest Accuracy : {0}'
    x_train,x_test,y_train,y_test=train_test(df)
    rfc.fit(x_train,y_train)
    y_pred=rfc.predict(x_test)
    
    Model_evaluation(title,y_test, y_pred)
    return y_pred
    
# çoklu sınıflandırma
def RandomForest_multi(df):
    tite='Random Forest Multi Classficiation Accuracy : {0}'
    x_train,x_test,y_train,y_test=train_test(df)
    mtf.fit(x_train, y_train.values.reshape(-1,1))
    y_pred=mtf.predict(x_test)
    
    Model_evaluation(tite,y_test,y_pred)
    return y_pred
    
def KNN(df):
    title='KNN Accuracy : {0}'
    x_train,x_test,y_train,y_test=train_test(df)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    Model_evaluation(title,y_test, y_pred)
    return y_pred

def LogisticRegression(df):
    title='Logistic Regression Accuracy : {0}'
    x_train,x_test,y_train,y_test=train_test(df)
    logr.fit(x_train,y_train)
    y_pred=logr.predict(x_test)
    Model_evaluation(title,y_test, y_pred)
    return y_pred

def NaiveBayes(df):
    title='Navie Bayes GaussianNB Accuracy : {0}'
    title2='Navie Bayes BernoulliNB Accuracy : {0}'
    title3='Navie Bayes MultinomialNB Accuracy : {0}'
    x_train,x_test,y_train,y_test=train_test(df)
    gnb.fit(x_train,y_train)
    gnb2.fit(x_train,y_train)
    gnb3.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    y_pred2=gnb2.predict(x_test)
    y_pred3=gnb3.predict(x_test)
    Model_evaluation(title,y_test, y_pred)
    Model_evaluation(title2,y_test, y_pred2)
    Model_evaluation(title3,y_test, y_pred3)
    return y_pred,y_pred2,y_pred3

def SVC(df):
    title='SVC Accuracy: {0}'
    x_train,x_test,y_train,y_test=train_test(df)
    svc.fit(x_train,y_train)
    y_pred=svc.predict(x_test)
    Model_evaluation(title,y_test, y_pred)
    return y_pred

def Model_evaluation(title,y_test,y_pred):
    print(title.format(accuracy_score(y_test, y_pred)))
    print("Confusion Matrix \n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
