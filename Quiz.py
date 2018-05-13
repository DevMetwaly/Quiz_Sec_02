"""
Created on Mon Apr  2 05:59:20 2018

@author: ABDO HAMDY METWALY
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler, Normalizer, FunctionTransformer
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"titanic.csv",index_col=0,na_values='',sep=',')
null_=['Age']
null_cat=['Pclass','Sex','SibSp','Ticket','Cabin','Embarked','Age']

#impute nulls
def impute_null(df,cols,cat):
    for col in cols:
        if(cat):
            df[col].fillna(df[col].mode()[0],inplace=True)
        else:
            df[col].fillna(df[col].mean(),inplace=True)
            
impute_null(df,null_cat,True)
impute_null(df,null_,False)


le = LabelEncoder()
for i in df.columns:
    if(df[i].dtype == "object"):
        df[i]=le.fit_transform(df[i])
        
X=df.iloc[:,1:]
Y=df.iloc[:,0]




def fn(X):
    poly = PolynomialFeatures(degree=6)
    X=poly.fit_transform(X)
    
    MinMax=MinMaxScaler(feature_range=(-1,1))
    
    X=MinMax.fit_transform(X) 
    
    
    nor = StandardScaler()
    X=nor.fit_transform(X)
    
    
    
    #res=poly.get_feature_names
    
    
    sel=SelectPercentile(score_func=f_classif,percentile=90)
    sel.fit(X,Y)
    X=sel.transform(X)
   
    return X

cust = FunctionTransformer(func=fn)
X=cust.fit_transform(X)



skf=StratifiedKFold(n_splits=5)
y_pred1=np.zeros(Y.shape[0])

for train,test in skf.split(X,Y):
    x_train=X[train]
    x_test=X[test]
    y_train=Y[train+1]
    y_test=Y[test+1]
    
    logReg = LogisticRegression()
    logReg.fit(x_train,y_train)
    y_pred1[test]=logReg.predict(x_test)

print(accuracy_score(Y,y_pred1),"%")

