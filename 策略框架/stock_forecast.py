import pandas as pd
import datetime,tushare
import numpy as np
from math import ceil
from sklearn import  preprocessing,cross_validation , svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

df=tushare.get_k_data('002415',index=False,ktype='60',start='2005/1/1',end='2017/7/26')

print(df.tail()[['date','close']])
df.dropna()

df=df[['date','open' , 'high' , 'low' , 'close' , 'volume']]
date =  df['date']
# df['HL_PCT']=(df['high']-df['close']) / df['close']*100.0
# df['PCT_change']=(df['close']-df['open']) / df['open']*100.0
# df=df[['date','close','HL_PCT','PCT_change','volume']]
# df=df=df[['open' , 'high' , 'low' , 'close' , 'volume']]
forecast_out = 2

df['lable']=df['close'].shift(-forecast_out)
XX = df.drop(['date','lable'],1)
y = df['lable']
x = preprocessing.scale(XX)
xx = x[:-forecast_out]
yy = df['lable'].dropna()

x_train,x_test,y_train,y_test = cross_validation.train_test_split(xx,yy)
lr = svm.SVR()
lr.fit(x_train,y_train)
score=lr.score(x_test,y_test)
#forecast = lr.predict()
print(score)



# with open('D:\\jupyter\\stock price\\stock_price_forecast\\lr.pickle','wb') as f:
#     pickle.dump(lr,f)
#
#
# # pickle_in = open('D:\jupyter\stock price\stock_price_forecast\lr.pickle','rb')
# # lr = pickle.load(pickle_in)
# df['forecast']=lr.predict(x)
# for i in range(forecast_out):
#     df.loc[df.shape[0]+i] = np.nan
# df['forecast']=df['forecast'].shift(forecast_out)
# print(df.tail())
# df.index = df['date']
# df[['close','forecast']].to_csv('D:\\jupyter\\stock price\\399001.csv')





















