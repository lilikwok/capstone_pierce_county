import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn import metrics
# import the regressor   
# import export_graphviz 
from sklearn.tree import export_graphviz  
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.utils import check_array

def mean_absolute_percentage_error(y_test, y_pred):
    y_test = y_test.values
    y_test = y_test.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    y_test = check_array(y_test)
    y_pred = check_array(y_pred)
    return np.mean(np.abs((y_test - y_pred)/y_test)) * 100

df = pd.read_csv("m1_c3.csv", sep=',', header=0)

y = df['sale_price']

df['View_Quality'] = df['View_Quality'] .astype('category')
df['Waterfront_Type'] = df['Waterfront_Type'] .astype('category')
df['withInSewerImprovement'] = df['withInSewerImprovement'] .astype('category')
df['near_firestation'] = df['near_firestation'] .astype('category')
df['near_hospital'] = df['near_hospital'] .astype('category')
df['near_libraries'] = df['near_libraries'] .astype('category')
df['near_policestation'] = df['near_policestation'] .astype('category')
df['near_waterplants'] = df['near_waterplants'] .astype('category')
df['condition'] = df['condition'] .astype('category')
df['quality'] = df['quality'] .astype('category')
df['attic_finished_square_feet'] = df['attic_finished_square_feet'] .astype('category')
df['basement_square_feet'] = df['basement_square_feet'] .astype('category')
df['basement_finished_square_feet'] = df['basement_finished_square_feet'] .astype('category')
df['porch_square_feet'] = df['porch_square_feet'] .astype('category')
df['attached_garage_square_feet'] = df['attached_garage_square_feet'] .astype('category')
df['detached_garage_square_feet'] = df['detached_garage_square_feet'] .astype('category')
df['fireplaces'] = df['fireplaces'] .astype('category')
df['near_private_school'] = df['near_private_school'] .astype('category')
df['near_elementary_school'] = df['near_elementary_school'] .astype('category')
df['near_high_school'] = df['near_high_school'] .astype('category')
df['near_college'] = df['near_college'] .astype('category')
df['Crime_Num'] = df['Crime_Num'].fillna(0)


X = df[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'Crime_Num', 'withInSewerImprovement',
       'near_firestation', 'near_hospital', 'near_libraries',
       'near_policestation', 'near_waterplants', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built', 'near_private_school', 'near_elementary_school', 'near_college', 'near_high_school']]
   
       
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109) 

# create a regressor object 
regressor = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=6,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
           
# fit the regressor with X and Y data 
rrtree = regressor.fit(X_train, y_train)  
y_pred = rrtree.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of random forest:",MSE)
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE: ", MAPE)

"""
export_graphviz(regressor.estimators_[0], out_file='tree_from_forest1.dot',feature_names=X.columns)
export_graphviz(regressor.estimators_[1], out_file='tree_from_forest2.dot',feature_names=X.columns)
export_graphviz(regressor.estimators_[2], out_file='tree_from_forest3.dot',feature_names=X.columns)
"""

# REFRENCE: https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/
# MSE:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html