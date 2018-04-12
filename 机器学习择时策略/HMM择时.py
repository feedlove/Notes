#HMM择时
"""
    策略思想：先选取沪深300中所有股票从2012/01/01-2015/01/01之间交
    易日的一日对数收益率，五日对数收益率，和五日成交量对数差作为观
    察序列属性，假设这些属性值服从正太高斯分布（HMM的强假设），我们
    直接利用这些历史数据完成HMM的构造，并且计算出其中表现最好的两个
    状态作为买入状态，其中表现最差的两个状态作为卖出状态，然后在回
    测过程中将当前回测日的特征值和之前训练集特征数据构成一个可观察
    序列，放入HMM中进行状态序列的预测，每个回测日取出前一日的状态，
    如果此状态在买入状态中则进行买入沪深300，如果此状态在卖出状态中
    而且我们有持仓则卖出沪深300。
    
    初始资金：1000000
    回测频率：每天
    回测日期：2015-08-01——2017-05-31
"""
#导入HMM模型所需的包0
from hmmlearn.hmm import GaussianHMM

import datetime
import numpy as np

#导入画图所需要的包
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import seaborn as sns

def HMMModel(stklist, start, end, n):
    """获取训练好的HMM模型以及买入和卖出状态
    
    参数
    ----------
    stklist : 列表
        训练数据包含的股票代码.
        
    start : '%Y%m%d'形式字符串
        训练数据的起始日期.
    
    end : '%Y%m%d'形式字符串
        训练数据的结束日期.
        
    n ：整数类型
        HMM模型中隐含的状态数.
    
    返回值
    ----------
    model : HMM模型
        训练好的HMM模型.  
    
    buystates : 列表
        买入的状态
    
    sellstates : 列表
        卖出的状态
        
    feature : 数组
        训练样本的特征数据集
    """
    df = get_price(stklist, start, end, '1d', ['close', 'volume'], True)
    
    #计算单日对数收益率
    df['1return'] = np.log(df['close']).diff(1)
    
    #计算五日对数收益率
    df['5return'] = np.log(df['close']).diff(5)
    
    #计算五日交易量对数差
    df['volume'] = np.log(df['volume'])
    df['5volume'] = df['volume'].diff(5)
    
    df = df.dropna()
    x = df['close']
    del df['close']
    
    #选取单日对数收益率，五日对数收益率，以及五日交易量对数差作为观测序列的特征因子
    feature = df.iloc[:,-3:].values
    
    #根据训练数据对HMM模型进行拟合
    model = GaussianHMM(n_components= n, covariance_type='diag', n_iter=2000).fit(feature)
    
    #通过训练数据计算不同隐含状态的累积收益率，选取前三作为买入状态后三个作为卖出状态
    hidden_states = model.predict(feature)
    x = pd.DataFrame(x)
    for i in range(n):
        x['%s'%i] = x['close']
        x['%s'%i] = 1
    for i in range(n):
        for j in range(0,len(x)-1):
            if hidden_states[j] == i:
                x.iloc[j+1,i+1] = (x.iloc[j+1,0]/x.iloc[j,0])*x.iloc[j,i+1]
            else:
                x.iloc[j+1,i+1] = x.iloc[j,i+1]
    ans = x.iloc[-1,1:]
    ans = ans.sort_values()
    buystates = ans.index.values[-2:]
    sellstates = ans.index.values[:2]
    for i in range(len(buystates)):
        buystates[i] = int(buystates[i])
        sellstates[i] = int(sellstates[i])
    return model, buystates, sellstates, feature

#初始化账户       
def initialize(account):      
    
    #设置要交易的证券(000300.SH 沪深300)      
    account.security = '000300.SH' 
    
    #初始化全局变量
    account.clf, account.buy, account.sell, account.feature  = HMMModel(account.security, '20120101', '20150101', 6)

#设置买卖条件，每个交易频率（日/分钟/tick）调用一次   
def handle_data(account,data):

    #计算测试集特征数据
    testdf = data.attribute_history(account.security, ['close','volume'], 10, '1d', True)
    testdf['1return'] = np.log(testdf['close']).diff(1)
    testdf['5return'] = np.log(testdf['close']).diff(5)
    testdf['volume'] = np.log(testdf['volume'])
    testdf['5volume'] = testdf['volume'].diff(5)
    testdf = testdf.dropna()
    
    #得到回测日前一日的特征数据
    feature_test = testdf.iloc[-1,-3:].values
    feature_test = feature_test.reshape(1,3)
    
    #将特征数据和之前的训练样本数据集结合起来当做观察序列
    account.feature = np.concatenate((account.feature,feature_test))

    if (account.clf.predict(account.feature)[-1] in account.buy):
        order_value(account.security,account.cash)
    if account.positions_value > 0:
        if (account.clf.predict(account.feature)[-1] not in account.buy):
            order_target(account.security, 0)
    