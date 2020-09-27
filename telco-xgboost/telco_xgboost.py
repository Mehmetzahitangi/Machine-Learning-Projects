# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:27:42 2020

@author: MehmetZahit
"""
#%%
import pandas as pd #loan and manipulate data and for one-hot encoding
import numpy as np #calculate the mean and standard deviation
import xgboost as xgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#import the data

dataFrame = pd.read_csv("Telco_customer_churn.csv")
print(dataFrame.head())

# the last four features contain exit interview information and should not be used for predict, we will remove them
dataFrame.drop(["Churn Label", "Churn Score","CLTV","Churn Reason"],axis=1,inplace=True)#axis=1 to remove columns
print(dataFrame.head())

#some of other features only contain a single value, so will not be useful for classification 
print(dataFrame["Count"].unique())
print(dataFrame["Country"].unique())
print(dataFrame["State"].unique())
#also remove CustomerID because it is different for every customer, so it is usefull for classification
#remove "Lat long" because there are separate columns for "latitude" and "longtitude"   
dataFrame.drop(["Count","Country","State","CustomerID","Lat Long"],axis=1,inplace=True)
print(dataFrame.head())

# if we want to draw a tree, we can't have any whitespace like in "Los Angeles" in city column
dataFrame["City"].replace(" ","_",regex=True, inplace=True) # convert " " to "_"
print(dataFrame.head())
# also eliminate the whitespaces in column names
dataFrame.columns = dataFrame.columns.str.replace(" ","_")

#MISSING DATA PART 1: identifying missing data
print(dataFrame.dtypes)

#MISSING DATA PART 1: dealing with missing data, XGBoost style
#convert to total_charges variables to zero and then it still will be object type
#but XGBoost only allows int,float or boolean types. we can fix this by converting it with to_numeric()

print(len(dataFrame.loc[dataFrame["Total_Charges"]==" "]))
dataFrame.loc[dataFrame["Total_Charges"]== " "]
dataFrame.loc[(dataFrame["Total_Charges"]==" "),"Total_Charges"] = 0
dataFrame.loc[dataFrame["Tenure_Months"]==0]

dataFrame["Total_Charges"] = pd.to_numeric(dataFrame["Total_Charges"])
print(dataFrame.dtypes)

dataFrame.replace(" ","_",regex=True,inplace=True)
print(dataFrame.head())

y = dataFrame["Churn_Value"]
features = dataFrame.drop("Churn_Value",axis=1)

#
print(pd.get_dummies(features, columns=["Payment_Method"]).head())

features_encoded = pd.get_dummies(features,columns=["City","Gender","Senior_Citizen","Partner","Dependents","Phone_Service",
                                                    "Multiple_Lines","Internet_Service","Online_Security","Online_Backup",
                                                    "Device_Protection","Tech_Support","Streaming_TV","Streaming_Movies",
                                                    "Contract","Paperless_Billing","Payment_Method"])
print(features_encoded.head())


#BUILD A PRELIMINARY XGBoost Model

features_train , features_test, y_train, y_test = train_test_split(features_encoded,y,random_state=42,stratify=y)

clf_xgb = xgb.XGBClassifier(objective="binary:logistic",missing=None,seed=42)
clf_xgb.fit(features_train,y_train,verbose=True,early_stopping_rounds=10,eval_metric="aucpr",eval_set=[(features_test,y_test)])

plot_confusion_matrix(clf_xgb,features_test, y_test,values_format="d",display_labels=["Did not leave","Left"])

# OPTIMIZING

param_grid = {"max_depth":[3,4,5],"learning_rate":[0.1,0.01,0.05],"gamma":[0,0.25,1.0],"reg_lambda":[0,1.0,10.0],
              "scale_pos_weight":[1,3,5]}
#best metrics 4,0.1,0.25,10,3

# final building
clf_xgb = xgb.XGBClassifier(objective="binary:logistic",missing=None,seed=42,gamma=0.25,learning_rate=0.1,max_depth=4,
                            reg_lambda=10,scale_pos_weight=3,subsample=0.9,colsample_bytree=0.5)

clf_xgb.fit(features_train,y_train,verbose=True,early_stopping_rounds=10,eval_metric="aucpr",eval_set=[(features_test,y_test)])

plot_confusion_matrix(clf_xgb,features_test, y_test,values_format="d",display_labels=["Did not leave","Left"])

clf_xgb = xgb.XGBClassifier(objective="binary:logistic",missing=None,seed=42,gamma=0.25,learning_rate=0.1,max_depth=4,
                            reg_lambda=10,scale_pos_weight=3,subsample=0.9,colsample_bytree=0.5,n_estimators=1) 
        #we set n_estimators to 1 so we can get gain,cover etc
clf_xgb.fit(features_train,y_train)

boost = clf_xgb.get_booster()

for importance_type in ("weight","gain","cover","total_gain","total_cover"):
    print("%s:" % importance_type, boost.get_score(importance_type=importance_type))

node_params = {"shape" : "box",#make the nodes fancy
               "style" : "filled,rounded",
               "fillcolor" : "#78cbe"} 

leaf_params = {"shape" : "box",#make the nodes fancy
               "style" : "filled",
               "fillcolor" : "#e48038"}



##if we want to save figure 
graph_data = xgb.to_graphviz(clf_xgb, num_trees=0,size="10,10",condition_node_params=node_params,leaf_node_params=leaf_params) 
graph_data.view(filename="xgboost_tree_customer_churn") # save as pdf


y_predict = clf_xgb.predict(features_test)
from sklearn.metrics import accuracy_score
print("Accuracy Score: ",accuracy_score(y_test, y_predict))
from sklearn.metrics import classification_report
print("Classification Report: ",classification_report(y_test, y_predict))
 
