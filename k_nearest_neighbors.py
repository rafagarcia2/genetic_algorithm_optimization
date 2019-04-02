import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/Train_UWu5bXk.csv')
# print(df.head())

#Limpando
df.isnull().sum()
mean = df['Item_Weight'].mean()
df['Item_Weight'].fillna(mean, inplace=True)

mode = df['Outlet_Size'].mode()
df['Outlet_Size'].fillna(mode[0], inplace=True)

df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1, inplace=True)
df = pd.get_dummies(df)

#Modelo/treino
train , test = train_test_split(df, test_size = 0.3)

x_train = train.drop('Item_Outlet_Sales', axis=1)
y_train = train['Item_Outlet_Sales']

x_test = test.drop('Item_Outlet_Sales', axis = 1)
y_test = test['Item_Outlet_Sales']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

# Genetic Algorthm

import genetic_algorithm_knn as GA

error = GA.cal_pop_fitness(range(1, 20), x_train, y_train, x_test, y_test)

for i in range(19):
    print('RMSE value for k = ' , i+1 , 'is:', error[i])