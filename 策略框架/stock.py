import pandas as pd
from pandas import Series,DataFrame
import numpy as np
  
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
# %matplotlib inline
  
import pandas_datareader as pdr
from datetime import datetime
# from __future__ import division

###股票代码
stock_lis = ["AAPL","GOOG","MSFT","AMZN"]
 
###开始及结束时间，这里我们去最近一年的数据 
end = datetime.now()
start = datetime(end.year - 5,end.month,end.day)
 
###将每个股票的近5年行情遍历出来 
# for stock in stock_lis:
#     globals()[stock] = pdr.get_data_yahoo(stock,"yahoo",start,end)
globals()["AAPL"]=pdr.get_data_yahoo("AAPL","yahoo",start,end)
print(AAPL.head())
