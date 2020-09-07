import os
import numpy as np
import pandas as pd

import  matplotlib
import  seaborn
import matplotlib.pyplot as plt

from sklearn import datasets

with open('./housingPrice/housing_data.txt', 'w') as new_file:
    with open('./housingPrice/housing.data.txt') as lines:
        for line in lines:
            new_eles = []
            eles = line.strip().split(' ')
            for i in eles:
                if not i :
                    continue
                new_eles.append(i)
            if len(new_eles) != 14: # 14 attitudes
                continue
            print(','.join(new_eles), file=new_file)

#print(len(new_eles))

col_names = []
with open('./housingPrice/readme.txt.txt', encoding='utf-8') as lines:
    for line in lines:
        line = line.rstrip().split(' ')[0]
        col_names.append(line)


#data
housing = pd.read_csv('./housingPrice/housing_data.txt', header=None, names=col_names)

print(housing)

# read target
target = housing["MEDV"].values

housing.info() #check

# 特徵縮放
#zcore minmax
# x = (X - Xmin) / (Xmax - Xmin)

from sklearn.preprocessing import  MinMaxScaler
minMax_scaler = MinMaxScaler()
minMax_scaler.fit(housing)
scaler_housing = minMax_scaler.transform(housing)
scaler_housing = pd.DataFrame(scaler_housing, columns=housing.columns)

from sklearn.linear_model import  LinearRegression
LR_reg = LinearRegression()
LR_reg.fit(scaler_housing, target)

#使用均方誤差
from  sklearn.metrics import  mean_squared_error
preds=LR_reg.predict(scaler_housing)
mse = mean_squared_error(preds, target) #使用均方誤差評價模型好壞,

#繪圖
plt.figure(figsize=(10, 7)) # size
num=100
x=np.arange(1, num+1)  # 取一百個值比較
plt.plot(x, target[:num], label='target') # target value
plt.plot(x, preds[:num], label='preds')
plt.legend(loc="upper right") # "線條顯示位置"
plt.show()
