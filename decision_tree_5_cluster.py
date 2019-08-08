import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn import metrics
from sklearn import tree
# import the regressor 
from sklearn.tree import DecisionTreeRegressor  
# import export_graphviz 
from sklearn.tree import export_graphviz  
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.utils import check_array

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true = y_true.values
    y_true = y_true.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    y_true = check_array(y_true)
    y_pred = check_array(y_pred)
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,2)
#########################################################
########## load data                             ########
#########################################################
m1_c1 = pd.read_csv("m2_c5_kmean-5clusters\m1_c1_kmean.csv", sep=',', header=0)
m1_c2 = pd.read_csv("m2_c5_kmean-5clusters\m1_c2_kmean.csv", sep=',', header=0)
m1_c3 = pd.read_csv("m2_c5_kmean-5clusters\m1_c3_kmean.csv", sep=',', header=0)
m1_c4 = pd.read_csv("m2_c5_kmean-5clusters\m1_c4_kmean.csv", sep=',', header=0)
m1_c5 = pd.read_csv("m2_c5_kmean-5clusters\m1_c5_kmean.csv", sep=',', header=0)
m1_full = pd.read_csv("m1_full.csv", sep=',', header=0)
m2_c1 = pd.read_csv("m2_c5_kmean-5clusters\m2_c1_kmean.csv", sep=',', header=0)
m2_c2 = pd.read_csv("m2_c5_kmean-5clusters\m2_c2_kmean.csv", sep=',', header=0)
m2_c3 = pd.read_csv("m2_c5_kmean-5clusters\m2_c3_kmean.csv", sep=',', header=0)
m2_c4 = pd.read_csv("m2_c5_kmean-5clusters\m2_c4_kmean.csv", sep=',', header=0)
m2_c5 = pd.read_csv("m2_c5_kmean-5clusters\m2_c5_kmean.csv", sep=',', header=0)
m2_full = pd.read_csv("m2_full.csv", sep=',', header=0)
print(len(m1_c1))
print(len(m1_c2))
print(len(m1_c3))
print(len(m1_c4))
print(len(m1_c5))
print(len(m1_full))
print(len(m2_c1))
print(len(m2_c2))
print(len(m2_c3))
print(len(m2_c4))
print(len(m2_c5))
print(len(m2_full))



##################################################################################
########## m1_c1,                                                    #############
########## MSE of m1_full: 5339046488.4728775, MAPE of m1_full 19.55 % ###########
##################################################################################

y = m1_c1['sale_price']

m1_c1['View_Quality'] = m1_c1['View_Quality'] .astype('category')
m1_c1['Waterfront_Type'] = m1_c1['Waterfront_Type'] .astype('category')
m1_c1['withInSewerImprovement'] = m1_c1['withInSewerImprovement'] .astype('category')
m1_c1['near_firestation'] = m1_c1['near_firestation'] .astype('category')
m1_c1['near_hospital'] = m1_c1['near_hospital'] .astype('category')
m1_c1['near_libraries'] = m1_c1['near_libraries'] .astype('category')
m1_c1['near_policestation'] = m1_c1['near_policestation'] .astype('category')
m1_c1['near_waterplants'] = m1_c1['near_waterplants'] .astype('category')
m1_c1['condition'] = m1_c1['condition'] .astype('category')
m1_c1['quality'] = m1_c1['quality'] .astype('category')
m1_c1['attic_finished_square_feet'] = m1_c1['attic_finished_square_feet'] .astype('category')
m1_c1['basement_square_feet'] = m1_c1['basement_square_feet'] .astype('category')
m1_c1['basement_finished_square_feet'] = m1_c1['basement_finished_square_feet'] .astype('category')
m1_c1['porch_square_feet'] = m1_c1['porch_square_feet'] .astype('category')
m1_c1['attached_garage_square_feet'] = m1_c1['attached_garage_square_feet'] .astype('category')
m1_c1['detached_garage_square_feet'] = m1_c1['detached_garage_square_feet'] .astype('category')
m1_c1['fireplaces'] = m1_c1['fireplaces'] .astype('category')
m1_c1['near_private_school'] = m1_c1['near_private_school'] .astype('category')
m1_c1['near_elementary_school'] = m1_c1['near_elementary_school'] .astype('category')
m1_c1['near_high_school'] = m1_c1['near_high_school'] .astype('category')
m1_c1['near_college'] = m1_c1['near_college'] .astype('category')
m1_c1['Crime_Num'] = m1_c1['Crime_Num'].fillna(0)


X = m1_c1[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'Crime_Num', 'withInSewerImprovement',
       'near_firestation', 'near_hospital', 'near_libraries',
       'near_policestation', 'near_waterplants', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built', 'near_private_school', 'near_elementary_school', 'near_college', 'near_high_school']]

###########  decison tree regression ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of m1_c1:\t",MSE)
print('MAPE of m1_c1:\t',mean_absolute_percentage_error(y_test, y_pred),'%')
# export the decision tree to a tree_5cluster.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree_m1_c1_5cluster.dot',feature_names=X.columns)  

##################################################################################
########## m1_c2,                                                    #############
########## MSE of m1_c2: 19872832122.125637, MAPE of m1_c2 19.55 %  ###########
##################################################################################

y = m1_c2['sale_price']

m1_c2['View_Quality'] = m1_c2['View_Quality'] .astype('category')
m1_c2['Waterfront_Type'] = m1_c2['Waterfront_Type'] .astype('category')
m1_c2['withInSewerImprovement'] = m1_c2['withInSewerImprovement'] .astype('category')
m1_c2['near_firestation'] = m1_c2['near_firestation'] .astype('category')
m1_c2['near_hospital'] = m1_c2['near_hospital'] .astype('category')
m1_c2['near_libraries'] = m1_c2['near_libraries'] .astype('category')
m1_c2['near_policestation'] = m1_c2['near_policestation'] .astype('category')
m1_c2['near_waterplants'] = m1_c2['near_waterplants'] .astype('category')
m1_c2['condition'] = m1_c2['condition'] .astype('category')
m1_c2['quality'] = m1_c2['quality'] .astype('category')
m1_c2['attic_finished_square_feet'] = m1_c2['attic_finished_square_feet'] .astype('category')
m1_c2['basement_square_feet'] = m1_c2['basement_square_feet'] .astype('category')
m1_c2['basement_finished_square_feet'] = m1_c2['basement_finished_square_feet'] .astype('category')
m1_c2['porch_square_feet'] = m1_c2['porch_square_feet'] .astype('category')
m1_c2['attached_garage_square_feet'] = m1_c2['attached_garage_square_feet'] .astype('category')
m1_c2['detached_garage_square_feet'] = m1_c2['detached_garage_square_feet'] .astype('category')
m1_c2['fireplaces'] = m1_c2['fireplaces'] .astype('category')
m1_c2['near_private_school'] = m1_c2['near_private_school'] .astype('category')
m1_c2['near_elementary_school'] = m1_c2['near_elementary_school'] .astype('category')
m1_c2['near_high_school'] = m1_c2['near_high_school'] .astype('category')
m1_c2['near_college'] = m1_c2['near_college'] .astype('category')
m1_c2['Crime_Num'] = m1_c2['Crime_Num'].fillna(0)


X = m1_c2[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'Crime_Num', 'withInSewerImprovement',
       'near_firestation', 'near_hospital', 'near_libraries',
       'near_policestation', 'near_waterplants', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built', 'near_private_school', 'near_elementary_school', 'near_college', 'near_high_school']]

###########  decison tree regression ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of m1_c2:\t",MSE)
print('MAPE of m1_c2:\t',mean_absolute_percentage_error(y_test, y_pred),'%')
# export the decision tree to a tree_5cluster.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree_m1_c2_5cluster.dot',feature_names=X.columns)  

##################################################################################
########## m1_c3,                                                    #############
########## MSE of m1_c3: 15026809238.01895, MAPE of m1_c3: 40.26 %     ###########
##################################################################################

y = m1_c3['sale_price']

m1_c3['View_Quality'] = m1_c3['View_Quality'] .astype('category')
m1_c3['Waterfront_Type'] = m1_c3['Waterfront_Type'] .astype('category')
m1_c3['withInSewerImprovement'] = m1_c3['withInSewerImprovement'] .astype('category')
m1_c3['near_firestation'] = m1_c3['near_firestation'] .astype('category')
m1_c3['near_hospital'] = m1_c3['near_hospital'] .astype('category')
m1_c3['near_libraries'] = m1_c3['near_libraries'] .astype('category')
m1_c3['near_policestation'] = m1_c3['near_policestation'] .astype('category')
m1_c3['near_waterplants'] = m1_c3['near_waterplants'] .astype('category')
m1_c3['condition'] = m1_c3['condition'] .astype('category')
m1_c3['quality'] = m1_c3['quality'] .astype('category')
m1_c3['attic_finished_square_feet'] = m1_c3['attic_finished_square_feet'] .astype('category')
m1_c3['basement_square_feet'] = m1_c3['basement_square_feet'] .astype('category')
m1_c3['basement_finished_square_feet'] = m1_c3['basement_finished_square_feet'] .astype('category')
m1_c3['porch_square_feet'] = m1_c3['porch_square_feet'] .astype('category')
m1_c3['attached_garage_square_feet'] = m1_c3['attached_garage_square_feet'] .astype('category')
m1_c3['detached_garage_square_feet'] = m1_c3['detached_garage_square_feet'] .astype('category')
m1_c3['fireplaces'] = m1_c3['fireplaces'] .astype('category')
m1_c3['near_private_school'] = m1_c3['near_private_school'] .astype('category')
m1_c3['near_elementary_school'] = m1_c3['near_elementary_school'] .astype('category')
m1_c3['near_high_school'] = m1_c3['near_high_school'] .astype('category')
m1_c3['near_college'] = m1_c3['near_college'] .astype('category')
m1_c3['Crime_Num'] = m1_c3['Crime_Num'].fillna(0)


X = m1_c3[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'Crime_Num', 'withInSewerImprovement',
       'near_firestation', 'near_hospital', 'near_libraries',
       'near_policestation', 'near_waterplants', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built', 'near_private_school', 'near_elementary_school', 'near_college', 'near_high_school']]
###########  decison tree regression ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of m1_c3:\t",MSE)
print('MAPE of m1_c3:\t',mean_absolute_percentage_error(y_test, y_pred),'%')
# export the decision tree to a tree_5cluster.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree_m1_c3_5cluster.dot',feature_names=X.columns)  


##################################################################################
########## m1_c4,                                                    #############
########## MSE of m1_c4: 15026809238.01895, MAPE of m1_c4: 40.26 %     ###########
##################################################################################

y = m1_c4['sale_price']

m1_c4['View_Quality'] = m1_c4['View_Quality'] .astype('category')
m1_c4['Waterfront_Type'] = m1_c4['Waterfront_Type'] .astype('category')
m1_c4['withInSewerImprovement'] = m1_c4['withInSewerImprovement'] .astype('category')
m1_c4['near_firestation'] = m1_c4['near_firestation'] .astype('category')
m1_c4['near_hospital'] = m1_c4['near_hospital'] .astype('category')
m1_c4['near_libraries'] = m1_c4['near_libraries'] .astype('category')
m1_c4['near_policestation'] = m1_c4['near_policestation'] .astype('category')
m1_c4['near_waterplants'] = m1_c4['near_waterplants'] .astype('category')
m1_c4['condition'] = m1_c4['condition'] .astype('category')
m1_c4['quality'] = m1_c4['quality'] .astype('category')
m1_c4['attic_finished_square_feet'] = m1_c4['attic_finished_square_feet'] .astype('category')
m1_c4['basement_square_feet'] = m1_c4['basement_square_feet'] .astype('category')
m1_c4['basement_finished_square_feet'] = m1_c4['basement_finished_square_feet'] .astype('category')
m1_c4['porch_square_feet'] = m1_c4['porch_square_feet'] .astype('category')
m1_c4['attached_garage_square_feet'] = m1_c4['attached_garage_square_feet'] .astype('category')
m1_c4['detached_garage_square_feet'] = m1_c4['detached_garage_square_feet'] .astype('category')
m1_c4['fireplaces'] = m1_c4['fireplaces'] .astype('category')
m1_c4['near_private_school'] = m1_c4['near_private_school'] .astype('category')
m1_c4['near_elementary_school'] = m1_c4['near_elementary_school'] .astype('category')
m1_c4['near_high_school'] = m1_c4['near_high_school'] .astype('category')
m1_c4['near_college'] = m1_c4['near_college'] .astype('category')
m1_c4['Crime_Num'] = m1_c4['Crime_Num'].fillna(0)

X = m1_c4[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'Crime_Num', 'withInSewerImprovement',
       'near_firestation', 'near_hospital', 'near_libraries',
       'near_policestation', 'near_waterplants', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built', 'near_private_school', 'near_elementary_school', 'near_college', 'near_high_school']]
###########  decison tree regression ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of m1_c4:\t",MSE)
print('MAPE of m1_c4:\t',mean_absolute_percentage_error(y_test, y_pred),'%')
# export the decision tree to a tree_5cluster.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree_m1_c4_5cluster.dot',feature_names=X.columns)  

##################################################################################
########## m1_c5,                                                    #############
########## MSE of m1_c5: 15026809238.01895, MAPE of m1_c5: 40.26 %     ###########
##################################################################################

y = m1_c5['sale_price']

m1_c5['View_Quality'] = m1_c5['View_Quality'] .astype('category')
m1_c5['Waterfront_Type'] = m1_c5['Waterfront_Type'] .astype('category')
m1_c5['withInSewerImprovement'] = m1_c5['withInSewerImprovement'] .astype('category')
m1_c5['near_firestation'] = m1_c5['near_firestation'] .astype('category')
m1_c5['near_hospital'] = m1_c5['near_hospital'] .astype('category')
m1_c5['near_libraries'] = m1_c5['near_libraries'] .astype('category')
m1_c5['near_policestation'] = m1_c5['near_policestation'] .astype('category')
m1_c5['near_waterplants'] = m1_c5['near_waterplants'] .astype('category')
m1_c5['condition'] = m1_c5['condition'] .astype('category')
m1_c5['quality'] = m1_c5['quality'] .astype('category')
m1_c5['attic_finished_square_feet'] = m1_c5['attic_finished_square_feet'] .astype('category')
m1_c5['basement_square_feet'] = m1_c5['basement_square_feet'] .astype('category')
m1_c5['basement_finished_square_feet'] = m1_c5['basement_finished_square_feet'] .astype('category')
m1_c5['porch_square_feet'] = m1_c5['porch_square_feet'] .astype('category')
m1_c5['attached_garage_square_feet'] = m1_c5['attached_garage_square_feet'] .astype('category')
m1_c5['detached_garage_square_feet'] = m1_c5['detached_garage_square_feet'] .astype('category')
m1_c5['fireplaces'] = m1_c5['fireplaces'] .astype('category')
m1_c5['near_private_school'] = m1_c5['near_private_school'] .astype('category')
m1_c5['near_elementary_school'] = m1_c5['near_elementary_school'] .astype('category')
m1_c5['near_high_school'] = m1_c5['near_high_school'] .astype('category')
m1_c5['near_college'] = m1_c5['near_college'] .astype('category')
m1_c5['Crime_Num'] = m1_c5['Crime_Num'].fillna(0)

X = m1_c5[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'Crime_Num', 'withInSewerImprovement',
       'near_firestation', 'near_hospital', 'near_libraries',
       'near_policestation', 'near_waterplants', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built', 'near_private_school', 'near_elementary_school', 'near_college', 'near_high_school']]
###########  decison tree regression ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of m1_c5:\t",MSE)
print('MAPE of m1_c5:\t',mean_absolute_percentage_error(y_test, y_pred),'%')
# export the decision tree to a tree_5cluster.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree_m1_c5_5cluster.dot',feature_names=X.columns)  


##################################################################################
########## m1_full,                                                  #############
########## MSE of m1_full: 5671768609.008166, MAPE of m1_full: 18.55 % ###########
##################################################################################

y = m1_full['sale_price']

m1_full['View_Quality'] = m1_full['View_Quality'] .astype('category')
m1_full['Waterfront_Type'] = m1_full['Waterfront_Type'] .astype('category')
m1_full['withInSewerImprovement'] = m1_full['withInSewerImprovement'] .astype('category')
m1_full['near_firestation'] = m1_full['near_firestation'] .astype('category')
m1_full['near_hospital'] = m1_full['near_hospital'] .astype('category')
m1_full['near_libraries'] = m1_full['near_libraries'] .astype('category')
m1_full['near_policestation'] = m1_full['near_policestation'] .astype('category')
m1_full['near_waterplants'] = m1_full['near_waterplants'] .astype('category')
m1_full['condition'] = m1_full['condition'] .astype('category')
m1_full['quality'] = m1_full['quality'] .astype('category')
m1_full['attic_finished_square_feet'] = m1_full['attic_finished_square_feet'] .astype('category')
m1_full['basement_square_feet'] = m1_full['basement_square_feet'] .astype('category')
m1_full['basement_finished_square_feet'] = m1_full['basement_finished_square_feet'] .astype('category')
m1_full['porch_square_feet'] = m1_full['porch_square_feet'] .astype('category')
m1_full['attached_garage_square_feet'] = m1_full['attached_garage_square_feet'] .astype('category')
m1_full['detached_garage_square_feet'] = m1_full['detached_garage_square_feet'] .astype('category')
m1_full['fireplaces'] = m1_full['fireplaces'] .astype('category')
m1_full['near_private_school'] = m1_full['near_private_school'] .astype('category')
m1_full['near_elementary_school'] = m1_full['near_elementary_school'] .astype('category')
m1_full['near_high_school'] = m1_full['near_high_school'] .astype('category')
m1_full['near_college'] = m1_full['near_college'] .astype('category')
m1_full['Crime_Num'] = m1_full['Crime_Num'].fillna(0)


X = m1_full[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'Crime_Num', 'withInSewerImprovement',
       'near_firestation', 'near_hospital', 'near_libraries',
       'near_policestation', 'near_waterplants', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built', 'near_private_school', 'near_elementary_school', 'near_college', 'near_high_school']]

###########  decison tree regression ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of m1_full:\t",MSE)
print('MAPE of m1_full:\t',mean_absolute_percentage_error(y_test, y_pred),'%')
# export the decision tree to a tree_5cluster.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree_m1_full_5cluster.dot',feature_names=X.columns)  





##############################################################################
########## m2_c1,                                                  ###########
########## MSE of m2_c1: 6108030579.272024, MAPE of m2_c1: 19.79 % ###########
##############################################################################
y = m2_c1['sale_price']

m2_c1['View_Quality'] = m2_c1['View_Quality'] .astype('category')
m2_c1['Waterfront_Type'] = m2_c1['Waterfront_Type'] .astype('category')
m2_c1['condition'] = m2_c1['condition'] .astype('category')
m2_c1['quality'] = m2_c1['quality'] .astype('category')
m2_c1['attic_finished_square_feet'] = m2_c1['attic_finished_square_feet'] .astype('category')
m2_c1['basement_square_feet'] = m2_c1['basement_square_feet'] .astype('category')
m2_c1['basement_finished_square_feet'] = m2_c1['basement_finished_square_feet'] .astype('category')
m2_c1['porch_square_feet'] = m2_c1['porch_square_feet'] .astype('category')
m2_c1['attached_garage_square_feet'] = m2_c1['attached_garage_square_feet'] .astype('category')
m2_c1['detached_garage_square_feet'] = m2_c1['detached_garage_square_feet'] .astype('category')
m2_c1['fireplaces'] = m2_c1['fireplaces'] .astype('category')

X = m2_c1[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built']] 

###########  decison tree regression ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of m2_c1:\t",MSE)
print('MAPE of m2_c1:\t',mean_absolute_percentage_error(y_test, y_pred),'%')
# export the decision tree to a tree_5cluster.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree_m2_c1_5cluster.dot',feature_names=X.columns)  


##############################################################################
########## m2_c2,                                                  ###########
########## MSE of m2_c2: 31678481493.910625, MAPE of m2_c2: 78.16 % ##########
##############################################################################
y = m2_c2['sale_price']

m2_c2['View_Quality'] = m2_c2['View_Quality'] .astype('category')
m2_c2['Waterfront_Type'] = m2_c2['Waterfront_Type'] .astype('category')
m2_c2['condition'] = m2_c2['condition'] .astype('category')
m2_c2['quality'] = m2_c2['quality'] .astype('category')
m2_c2['attic_finished_square_feet'] = m2_c2['attic_finished_square_feet'] .astype('category')
m2_c2['basement_square_feet'] = m2_c2['basement_square_feet'] .astype('category')
m2_c2['basement_finished_square_feet'] = m2_c2['basement_finished_square_feet'] .astype('category')
m2_c2['porch_square_feet'] = m2_c2['porch_square_feet'] .astype('category')
m2_c2['attached_garage_square_feet'] = m2_c2['attached_garage_square_feet'] .astype('category')
m2_c2['detached_garage_square_feet'] = m2_c2['detached_garage_square_feet'] .astype('category')
m2_c2['fireplaces'] = m2_c2['fireplaces'] .astype('category')

X = m2_c2[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built']] 

###########  decison tree regression ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of m2_c2:\t",MSE)
print('MAPE of m2_c2:\t',mean_absolute_percentage_error(y_test, y_pred),'%')
# export the decision tree to a tree_5cluster.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree_m2_c2_5cluster.dot',feature_names=X.columns)  

##############################################################################
########## m2_c3,                                                  ###########
########## MSE of m2_c3: 25338970232.225716, MAPE of m2_c3: 78.68 % ##########
##############################################################################
y = m2_c3['sale_price']

m2_c3['View_Quality'] = m2_c3['View_Quality'] .astype('category')
m2_c3['Waterfront_Type'] = m2_c3['Waterfront_Type'] .astype('category')
m2_c3['condition'] = m2_c3['condition'] .astype('category')
m2_c3['quality'] = m2_c3['quality'] .astype('category')
m2_c3['attic_finished_square_feet'] = m2_c3['attic_finished_square_feet'] .astype('category')
m2_c3['basement_square_feet'] = m2_c3['basement_square_feet'] .astype('category')
m2_c3['basement_finished_square_feet'] = m2_c3['basement_finished_square_feet'] .astype('category')
m2_c3['porch_square_feet'] = m2_c3['porch_square_feet'] .astype('category')
m2_c3['attached_garage_square_feet'] = m2_c3['attached_garage_square_feet'] .astype('category')
m2_c3['detached_garage_square_feet'] = m2_c3['detached_garage_square_feet'] .astype('category')
m2_c3['fireplaces'] = m2_c3['fireplaces'] .astype('category')

X = m2_c3[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built']] 
###########  decison tree regression ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of m2_c3:\t",MSE)
print('MAPE of m2_c3:\t',mean_absolute_percentage_error(y_test, y_pred),'%')
# export the decision tree to a tree_5cluster.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree_m2_c3_5cluster.dot',feature_names=X.columns)  


##############################################################################
########## m2_c4,                                                  ###########
########## MSE of m2_c4: 25338970232.225716, MAPE of m2_c4: 78.68 % ##########
##############################################################################
y = m2_c4['sale_price']

m2_c4['View_Quality'] = m2_c4['View_Quality'] .astype('category')
m2_c4['Waterfront_Type'] = m2_c4['Waterfront_Type'] .astype('category')
m2_c4['condition'] = m2_c4['condition'] .astype('category')
m2_c4['quality'] = m2_c4['quality'] .astype('category')
m2_c4['attic_finished_square_feet'] = m2_c4['attic_finished_square_feet'] .astype('category')
m2_c4['basement_square_feet'] = m2_c4['basement_square_feet'] .astype('category')
m2_c4['basement_finished_square_feet'] = m2_c4['basement_finished_square_feet'] .astype('category')
m2_c4['porch_square_feet'] = m2_c4['porch_square_feet'] .astype('category')
m2_c4['attached_garage_square_feet'] = m2_c4['attached_garage_square_feet'] .astype('category')
m2_c4['detached_garage_square_feet'] = m2_c4['detached_garage_square_feet'] .astype('category')
m2_c4['fireplaces'] = m2_c4['fireplaces'] .astype('category')

X = m2_c4[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built']] 
###########  decison tree regression ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of m2_c4:\t",MSE)
print('MAPE of m2_c4:\t',mean_absolute_percentage_error(y_test, y_pred),'%')
# export the decision tree to a tree_5cluster.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree_m2_c4_5cluster.dot',feature_names=X.columns)  



##############################################################################
########## m2_c5,                                                  ###########
########## MSE of m2_c5: 25338970232.225716, MAPE of m2_c5: 78.68 % ##########
##############################################################################
y = m2_c5['sale_price']

m2_c5['View_Quality'] = m2_c5['View_Quality'] .astype('category')
m2_c5['Waterfront_Type'] = m2_c5['Waterfront_Type'] .astype('category')
m2_c5['condition'] = m2_c5['condition'] .astype('category')
m2_c5['quality'] = m2_c5['quality'] .astype('category')
m2_c5['attic_finished_square_feet'] = m2_c5['attic_finished_square_feet'] .astype('category')
m2_c5['basement_square_feet'] = m2_c5['basement_square_feet'] .astype('category')
m2_c5['basement_finished_square_feet'] = m2_c5['basement_finished_square_feet'] .astype('category')
m2_c5['porch_square_feet'] = m2_c5['porch_square_feet'] .astype('category')
m2_c5['attached_garage_square_feet'] = m2_c5['attached_garage_square_feet'] .astype('category')
m2_c5['detached_garage_square_feet'] = m2_c5['detached_garage_square_feet'] .astype('category')
m2_c5['fireplaces'] = m2_c5['fireplaces'] .astype('category')

X = m2_c5[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built']] 
###########  decison tree regression ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of m2_c5:\t",MSE)
print('MAPE of m2_c5:\t',mean_absolute_percentage_error(y_test, y_pred),'%')
# export the decision tree to a tree_5cluster.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree_m2_c5_5cluster.dot',feature_names=X.columns)  




##################################################################################
########## m2_full,                                                  #############
########## MSE of m2_full: 5812328073.374844, MAPE of m2_full: 18.68 %  ##########
##################################################################################
y = m2_full['sale_price']

m2_full['View_Quality'] = m2_full['View_Quality'] .astype('category')
m2_full['Waterfront_Type'] = m2_full['Waterfront_Type'] .astype('category')
m2_full['condition'] = m2_full['condition'] .astype('category')
m2_full['quality'] = m2_full['quality'] .astype('category')
m2_full['attic_finished_square_feet'] = m2_full['attic_finished_square_feet'] .astype('category')
m2_full['basement_square_feet'] = m2_full['basement_square_feet'] .astype('category')
m2_full['basement_finished_square_feet'] = m2_full['basement_finished_square_feet'] .astype('category')
m2_full['porch_square_feet'] = m2_full['porch_square_feet'] .astype('category')
m2_full['attached_garage_square_feet'] = m2_full['attached_garage_square_feet'] .astype('category')
m2_full['detached_garage_square_feet'] = m2_full['detached_garage_square_feet'] .astype('category')
m2_full['fireplaces'] = m2_full['fireplaces'] .astype('category')

X = m2_full[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built']] 

###########  decison tree regression ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

# create a regressor object 
regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

# fit the regressor with X and Y data 
reg = regressor.fit(X_train, y_train) 
y_pred = reg.predict(X_test)  
MSE = mean_squared_error(y_test, y_pred)
print("MSE of m2_full:\t",MSE)
print('MAPE of m2_full:\t',mean_absolute_percentage_error(y_test, y_pred),'%')
# export the decision tree to a tree_5cluster.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree_m2_full_5cluster.dot',feature_names=X.columns)  


