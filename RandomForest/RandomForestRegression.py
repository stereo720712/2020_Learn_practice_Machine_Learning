import pandas

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from  sklearn import tree
from sklearn.datasets import  load_boston

#data
boston_house = load_boston()
boston_feature_name = boston_house.feature_names
boston_features = boston_house.data
boston_target = boston_house.target

print(boston_feature_name)
print('\n')
print(boston_house.DESCR)
print(boston_features[:5,:])

rgs = RandomForestRegressor(n_estimators=10)
rgs = rgs.fit(boston_features, boston_target)
print(rgs)
res = rgs.predict(boston_features)
print(res)
# compare to random tree
rgs2 = tree.DecisionTreeRegressor()
rgs2.fit(boston_features, boston_target)
print(rgs2.predict(boston_features))