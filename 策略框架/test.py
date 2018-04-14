from __future__ import  print_function
#coding=utf-8
#Version:python3.6.0
#tools:Pycharm 2017.3.2
__Date__ = '2018/2/19 0:04'
__Author__ = 'admin'

import datetime
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
df = pdr.get_data_yahoo("AAPL")
# df[["Close","Open"]].plot()
# plt.show()
print(ts.adfuller(df["Close"],1))
# print()
