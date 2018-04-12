# 基于小波变换的ARMA模型量化投资策略

"""
 策略思想：因为小波变换耗时较长，所以只选择一只股票进行研究。本文选取
  平安银行作为研究对象，对股票过去100个交易日价格序列进行小波变
  换分解得到各层小波系数，在小波基函数选择上，对称性是小波基函数
  在选取上要考虑的一个重要因素，对称性不好会造成原始信号在重构后
  有相移的存在。所以本文在小波基函数的选取上最终选择对称性好的小
  波函数db4，分解层数为2。利用ARMA模型对各层小波系数进行建模预测，
  将其小波系数的预测值重构生成1日后（向前一步）股票价格预测值。对
  于该预测值，如果它超过回测当日开盘价的1%且未持仓，那么就以当日
  开盘价买入，反之，如果预测值会小于当日的开盘价且已持有就卖出。
  
 初始资金：100000
 回测频率：每天
 回测日期：2016-03-01 — 2016-08-01

"""


import pandas as pd
import numpy as np
import pywt   
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from matplotlib import pyplot as plt

def predict_wavelet(data_array): 
    """ 获取向前一步的序列预测值
    参数
    ----------
    data_array : numpy array类型
        股票价格序列.
      
    返回值
    ----------
    ans : float类型
        向前一步预测值
    """
    A, B, C = pywt.wavedec(data_array,'db4',mode='sym',level=2)
    #利用AIC准则确定ARMA模型的阶
    order_A = sm.tsa.arma_order_select_ic(A, ic='aic')['aic_min_order']
    order_B = sm.tsa.arma_order_select_ic(B, ic='aic')['aic_min_order']
    order_C = sm.tsa.arma_order_select_ic(C, ic='aic')['aic_min_order']
    #确定阶数以后就构建ARMA模型
    model_A = ARMA(A, order = order_A)
    results_A = model_A.fit()
    model_B = ARMA(B, order = order_B)
    results_B = model_B.fit()
    model_C = ARMA(C, order = order_C)
    results_C = model_C.fit()
    #根据预测的步数确定delta
    lag = [0,0,1]  
    # 预测小波系数
    pre_A = model_A.predict(results_A.params, 1, len(A) + lag[0])
    pre_B = model_B.predict(results_B.params, 1, len(B) + lag[1])
    pre_C = model_C.predict(results_C.params, 1, len(C) + lag[2])
    # 利用预测的小波系数重构形成最终的预测值
    predict_array = pywt.waverec([pre_A,pre_B,pre_C],'db4')
    ans = predict_array[-1]
    return ans

 
def get_date_delta(current_date, delta):
    """获得当前交易日 + delta（可以为负）的交易日的日期
    参数
    ----------
    current_date : '%Y-%m-%d'形式字符串
        当前交易日的日期.
    
    delta ：整数类型
        想获取的日期差.
    
    返回值
    ----------
    ans : '%Y-%m-%d'形式字符串
        当前交易日 + delta（可以为负）的交易日的日期.     
    """
    date = list(get_trade_days('20050101', '20171231').strftime('%Y-%m-%d'))
    return date[date.index(current_date) + delta]


def initialize(account):
    account.security = '000001.SZ'
   
   
def handle_data(account,data):
    stock = account.security
    # 获取上一个交易日
    yesterday = get_last_datetime().strftime('%Y-%m-%d')
    # 获取过去100个交易日数据
    data_array = get_price([stock], start_date = get_date_delta(yesterday, -99), end_date = yesterday, fre_step = '1d', fields = ['open'], skip_paused=False, fq=None)
    data_array = data_array[stock]['open'].values
    try:
        # 预测向前一步的序列预测值
        pre_prices = predict_wavelet(data_array)
        # 如果预测上涨幅度大于1%就以今天的开盘价买入
        if pre_prices / data.current([stock])[stock].open - 1 >= 0.01 and stock not in account.positions.keys():
            log.info('buying %s' % stock)
            order_percent(stock, 0.85)
        # 如果预测价格会下跌且已持有就以今天开盘价卖出
        if pre_prices / data.current([stock])[stock].open - 1 <= 0 and stock in account.positions.keys():
            log.info('selling %s' % stock)
            order_target(stock, 0)
    except Exception as e:
        pass