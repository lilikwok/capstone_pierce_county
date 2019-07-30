import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn import metrics
from sklearn import tree
# import the regressor 
from sklearn.tree import DecisionTreeRegressor  
# import export_graphviz 
from sklearn.tree import export_graphviz  
from sklearn.metrics import mean_squared_error


df = pd.read_csv("model.csv", sep=',', header=0)

y = df['sale_price']
print(df.dtypes)
df['View_Quality'] = df['View_Quality'] .astype('category')
df['Waterfront_Type'] = df['Waterfront_Type'] .astype('category')
df['withInSewerImprovement'] = df['withInSewerImprovement'] .astype('category')
df['near_firestation'] = df['near_firestation'] .astype('category')
df['near_healthcare'] = df['near_healthcare'] .astype('category')
df['near_libraries'] = df['near_libraries'] .astype('category')
df['near_policestation'] = df['near_policestation'] .astype('category')
df['near_schools'] = df['near_schools'] .astype('category')
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
df['Crime_Num'] = df['Crime_Num'].fillna(0)
print(df.columns)

X = df[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'Crime_Num', 'withInSewerImprovement',
       'near_firestation', 'near_healthcare', 'near_libraries',
       'near_policestation', 'near_schools', 'near_waterplants', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=6, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE:",MSE)
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree.dot',feature_names=X.columns)  


# REFRENCE: https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/
# MSE:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

'''
 full model with tree_depth = 5 
Important factor : square_feet, basement_finished_square_feet, quality, 
Land_Net_Acres, year_built, bedrooms, bathrooms, 
fireplaces, porch_square_feet, Waterfront_Type, Crime_Num, Waterfront_Type
'''

X1 = df[['Land_Net_Acres', 
       'Waterfront_Type', 'Crime_Num', 
       'square_feet',
       'condition', 'quality', 'basement_finished_square_feet',
       'porch_square_feet',  'fireplaces', 'bedrooms',
       'bathrooms', 'year_built']]

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.20,random_state=109)  

# create a regressor object 

regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=5, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X1_train, y_train) 
y1_pred = reg.predict(X1_test)  
MSE1 = mean_squared_error(y_test, y1_pred)
print("MSE:",MSE1)
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree1.dot',feature_names=X1.columns)  