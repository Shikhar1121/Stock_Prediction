# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 00:11:01 2020

@author: Shikhar
"""
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df_train = pd .read_csv("Google_Stock_Price_Train.csv")
train_set = df_train.iloc[:,1:2].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train_set_scale = sc.fit_transform(train_set)
x_train= []
y_train=[]
for i in range(60,1258):
    x_train.append(train_set_scale[i-60:i,0])
    y_train.append(train_set_scale[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences=False))
regressor.add(Dropout(0.2))


regressor.add(Dense(units=1))

regressor.compile(optimizer='adam',loss='mean_squared_error')

regressor.fit(x_train,y_train,epochs=100,batch_size=32)

df_test = pd .read_csv("Google_Stock_Price_Test.csv")
test_set = df_test.iloc[:,1:2].values

df_totals = pd .concat((df_train['Open'],df_test['Open']),axis = 0)

inputs = df_totals[len(df_totals)-len(df_test)-60:].values

inputs = inputs.reshape(-1,1)
inputs= sc.transform(inputs)


x_test= []
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)

x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predicted_stock_price= regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(test_set,color='red',label= 'Real Google Stock price')
plt.plot(predicted_stock_price,color='green',label= 'Predicted Google Stock price')

plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.title('Google Stock Price')
plt.legend()
plt.show()