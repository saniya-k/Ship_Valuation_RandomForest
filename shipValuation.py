# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:28:15 2020

@author: Saniya
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#read the file
ship_data=pd.read_csv("data/RegressionData.csv")
bet_performer_data=pd.read_csv("data/testBP.csv")
#check data format
print(ship_data.head())

#select features for regression
ship_features=["Age_at_Sale","DWT","Capesize"]

#select the target variable column ie sale price
y=ship_data.Price

#select features
X=ship_data[ship_features]
X_test=bet_performer_data[ship_features]

#split into validation and training set
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=1)

#define model
model_RF=RandomForestRegressor(n_estimators=500,random_state=1)

#fit the model
model_RF.fit(train_X,train_y)

#make validation prediction and caluclate MAE
val_preds=model_RF.predict(val_X)
val_MAE=mean_absolute_error(val_preds,val_y)
print("Validation MAE in Random Forest: {:,.0f}".format(val_MAE))

#predict price of Bet performer
price_Bp=model_RF.predict(X_test)
#Print price of ship
print(price_Bp)

