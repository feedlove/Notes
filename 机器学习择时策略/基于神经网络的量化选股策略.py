# 基于神经网络的量化选股策略

"""
 策略思想：选取一些常用技术指标作为训练样本特征集，对于类别，如果未来
        20个工作日上涨幅度超过10%则标记为1，否则为-1，采用神经网络算法
        进行训练，预测结果为1且未持仓则买入，预测结果为-1且已持仓则卖出。

 特征因子选取：本文采用神经网络算法解决有监督学习的分类问题，特征因子选
        取了2015-01-05那天沪深300成份股的总市值，OBV能量潮，市盈率，布林线，
        KDJ随机指标，RSI相对强弱指标共6个指标。
 
 数据标准化：数据标准化方法有很多，本文采用高斯预处理方法，即每个特征因子
        减去它对应的均值再除以它的标准差((x-x.mean)/x.std)。
 
 初始资金：10000000
 回测频率：每天
 回测日期：2015-03-02 — 2017-03-01
 
"""

import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier

def get_factor_data(stock_list, date):
    """ 获取特征因子数据
    参数
    ----------
    stock_list : list类型
        股票列表.
      
    start_data : '%Y-%m-%d'形式字符串
        获取因子参考日期.
      
    返回值
    ----------
    ans : pandas DataFrame结构形式
         特征因子数据
    """
    # 因子对应名称参考网页： 'http://quant.10jqka.com.cn/platform/html/help-api.html?t=data#222/0'
    q = query(
            factor.symbol,
            factor.zdzb,
            factor.market_cap,
            factor.obv,
            factor.pe_ttm,
            factor.boll,
            factor.kdj,
            factor.rsi
        ).filter(
            factor.symbol.in_(stock_list),
            factor.date == date
            )
    ans = get_factors(q)
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
    ans = date[date.index(current_date) + delta]
    return ans 
    
    
def create_data(date, delta, stock_list):
    """ 数据集的生成 
    参数
    ----------
    start_data ： '%Y-%m-%d'形式字符串
        训练集选取开始日期
      
    delta ：int类型
         工作日间隔（参考get_date_delta函数）
      
    stock_list : list类型
        股票列表
      
    返回值
    ----------
    df.values ： numpy ndarray结构
        训练样本特征集
     
    label ： list类型
         类别
    """
    df = get_factor_data(stock_list, date)
    #在df数据集上生成label（类别）列，涨幅超过10%标为1，反之为-1，注意lambda函数的应用
    df['label'] = df['factor_symbol'].map(lambda x: 1 if get_price([x], start_date = date, end_date = get_date_delta(date, delta), fields = ['open'])[x].open[-1]/get_price([x], start_date = date, end_date = get_date_delta(date, delta), fields = ['open'])[x].open[0] - 1 >= 0.1 else -1)
    df = df.dropna()
    df = df.set_index('factor_symbol')
    label = list(df['label'])
    del(df['label'])
    return df.values, label      
    
    
def initialize(account):
    # 基准日期选取2015-01-05，回测对象：沪深300成分股
    account.date = '2015-01-05'
    account.security = get_index_stocks('000300.SH', account.date)
    # 获取训练集及类别
    df, label = create_data(account.date, 20, account.security)
    # 以下两行为数据标准化
    account.scaler = preprocessing.StandardScaler().fit(df)  
    df_scaler = account.scaler.transform(df)
     # 神经网络算法模型构建
    account.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    account.clf.fit(df_scaler, np.array(label))
    
    
def handle_data(account,data):  
    # 获取上个交易日日期    
    yesterday_date = get_last_datetime()
    # 以下三行获取上个交易日的特征因子数据
    factor_data = get_factor_data(account.security, yesterday_date.strftime('%Y-%m-%d'))
    result = factor_data['factor_symbol'].values
    del(factor_data['factor_symbol'])
     # 对预测值为1的股票都放入stocks_buy列表中
    stocks_buy = result[account.clf.predict(account.scaler.transform(factor_data.values)) == 1]
    for stock in stocks_buy:
        if stock not in account.positions.keys():
            log.info('buying %s' % stock)
            order_value(stock, account.cash / (len(stocks_buy)+1))
    # 对预测值为-1的股票都放入stocks_buy列表中        
    stocks_sell = result[account.clf.predict(account.scaler.transform(factor_data.values)) == -1]
    for stock in stocks_sell:
        if stock in account.positions.keys():
            log.info('selling %s' % stock)
            order_target(stock, 0)