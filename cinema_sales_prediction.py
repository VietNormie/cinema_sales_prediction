# 1. Import library 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
from sklearn.model_selection import cross_val_score 

# 2. Load the dataset and data exploration, data preprocessing  

data = pd.read_csv('data/cinemaTicket_Ref.csv')
data.head()

# Data imputation  

data.info()

data.isnull().sum()

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean") 
imputer.fit(data[['occu_perc', 'capacity']])
data[['occu_perc', 'capacity']] = imputer.transform(data[['occu_perc', 'capacity']])
data.isnull().sum()

# Encode categorical data 

data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['timestamp'] = data['date'].view('int64') / 10**9
data = data.drop(columns = ['date']) 

data.info()

data.head()

# Choose significant features 

plt.figure(figsize = (15, 15)) 
sns.heatmap(data.corr(), annot = True)

data.columns

features = ["tickets_sold", "show_time", "occu_perc", "ticket_price", "ticket_use", "capacity"]

X = data[features]
y = data["total_sales"]

# 3. Splitting the dataset and feature scaling  
     
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 0)  

X_train 

X_test

# Feature scaling 

sc1 = StandardScaler()
sc2 = StandardScaler() 

X_train[['ticket_price', 'capacity']] = sc1.fit_transform(X_train[['ticket_price', 'capacity']])
X_test[['ticket_price', 'capacity']] = sc1.transform(X_test[['ticket_price', 'capacity']])
X_train[['tickets_sold', 'ticket_use']] = sc2.fit_transform(X_train[['tickets_sold', 'ticket_use']])
X_test[['tickets_sold', 'ticket_use']] = sc2.transform(X_test[['tickets_sold', 'ticket_use']]) 

X_train 

X_test 

# 4. Build the model 

# Linear Regression 

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

res = r2_score(y_test, y_pred)
print("r_2 score: ", res) 

# Decision Tree Regressor 

dtr_model = DecisionTreeRegressor(random_state = 1)
dtr_model.fit(X_train,y_train)
y_pred = dtr_model.predict(X_test)

res = r2_score(y_test, y_pred)
print("r_2 score: ", res) 

# Random Forest Regressor 

rfr_model = RandomForestRegressor()
rfr_model.fit(X_train, y_train)
y_pred = rfr_model.predict(X_test)

res = r2_score(y_test, y_pred)
print("r_2 score: ", res) 

# XGBoost Regressor 

xgb_model = XGBRegressor() 
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

res = r2_score(y_test, y_pred)
print("r_2 score: ", res) 

# 5. Final model evaluation 
 
model = XGBRegressor() 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

res = r2_score(y_test, y_pred)
print("r_2 score: ", res) 

error1 = root_mean_squared_error(y_test, y_pred)
print("Root mean squared error: ", error1)

error2 = mean_absolute_percentage_error(y_test, y_pred)
print("Mean absolute percentage error: ", error2)

score = cross_val_score(model, X, y, cv = 5, scoring = 'r2')
print('Cross val score: ', score, score.mean(), score.std()) 

plt.scatter(y_test, y_pred, alpha = 0.5)

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r', linestyle = '--')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted values')    
plt.show() 


