# 逻辑回归基于多因子选股
"""
    策略思想：针对沪深300（000300.SH 沪深300）中所有股票，选出7个因子
    （zdzb：ZDZB筑底指标,market_cap:总市值 ,obv:OBV能量潮 ,pe_ttm:市
    盈率TTM ,boll:BOLL布林线 ,pb:市净率 ,kdj:KDJ随机指标)作为训练样本
    特征属性，然后对于每一个回测日，取出回测日前四十一个交易日的沪深300
    所有股票这七个因子进行标准化操作作为训练集特征值，对于类别值以20天
    为基准，20日后涨幅超过阀值（2%）为1，反之为-1，然后利用逻辑回归对训
    练集进行训练，然后利用训练好的模型对当前回测日的前一个交易日的这七
    个因子数据先进性标准化再进行类别概率预测，并且调整调仓周期为20天，每
    个调仓日选出类别为1概率最大的三只股票，如果当前持仓股票不在这三只股
    票当中则全部卖出，然后等权买入三只股票。
    
    初始资金：100000
    回测频率：每天
    回测日期：2016-04-01——2017-04-01
"""
import numpy as np
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation
import datetime

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

def get_factor_data(stock_list, date):
    """获取特定日期的特征因子数据
    
    参数
    ----------
    stock_list : 列表
        股票列表.
    
    date ：'%Y-%m-%d'形式字符串
        获取因子日期.
    
    返回值
    ----------
    get_factors(q) : DataFrame
        以DataFrame的形式获取特定日期的特征因子数据     
    """
    q = query(
            factor.symbol,
            factor.zdzb,
            factor.market_cap,
            factor.obv,
            factor.pe_ttm,
            factor.boll,
            factor.pb,
            factor.kdj
        ).filter(
            factor.symbol.in_(stock_list),
            factor.date == date
            )
    return get_factors(q)

def Logisticdata(stklist, start, end):
    """获取逻辑回归的训练数据特征值以及相应的类别
    
    参数
    ----------
    stklist : 列表
        股票列表.
    
    start ：'%Y%m%d'形式字符串
        获取训练数据的起始日期.
    
    end : '%Y%m%d'形式字符串
        获取训练数据的结束日期.
    
    返回值
    ----------
    feature : 数组类型
        获取的训练集中的特征值数据
    
    lable ： 数组类型
        获取的训练集中的类别数据
    """
    Data_set = pd.DataFrame()
    for traindate in get_trade_days(start, end):
        traindate = traindate.strftime('%Y-%m-%d')
        df = get_factor_data(stklist, traindate)
        #log.info(df)
        df['label'] = df['factor_symbol'].map(lambda x: 1 if get_price([x], start_date = get_date_delta(traindate,1), end_date = get_date_delta(traindate,21), fields = ['open'])[x].open[-1]/get_price([x], start_date = get_date_delta(traindate,1), end_date = get_date_delta(traindate,21), fields = ['open'])[x].open[0] - 1 >= 0.02 else -1)
        df = df.set_index('factor_symbol')
        df = df.dropna() 
    Data_set = pd.concat([Data_set,df], axis = 0)
    feature = Data_set.iloc[:,:-1].values
    lable = Data_set['label'].T.values
    return feature, lable 
    
def listget(stklist, t, account):
    """获取每日要买入的股票列表
    
    参数
    ----------
    stklist : 列表
        股票列表.
    
    t ：日期格式
        回测日前一个交易日的日期.
    
    account : 包含initialize中定义的全局变量
        股票账号信息.
    
    返回值
    ----------
    buy : 列表
        要买入的股票列表
    """
    q = query(
        factor.symbol,
        factor.zdzb,
        factor.market_cap,
        factor.obv,
        factor.pe_ttm,
        factor.boll,
        factor.pb,
        factor.kdj
    ).filter(
        factor.symbol.in_(stklist),
        factor.date==t.strftime('%Y-%m-%d')
    )
    tdf = get_factors(q).set_index('factor_symbol').dropna()
    buy = []
    if len(tdf) > 0:
        
        feature_test = account.nm.transform(tdf.values)
        lable_test = account.clf.predict_proba(feature_test)[:,1]
        ans = pd.DataFrame(lable_test, index = tdf.index.values)
        buy = ans.sort_values(by = 0).index.values[-3:]
    return buy
        
#初始化账户       
def initialize(account):      
    
    #设置要交易的证券(沪深300)      
    account.security = get_index_stocks('000300.SH', '20150105') 
    
    set_benchmark('000300.SH')
    account.clf = LogisticRegression()
    account.nm = MinMaxScaler()
    account.count = -1
    
#设置买卖条件，每个交易频率（日/分钟/tick）调用一次   
def handle_data(account,data):
    account.count += 1
    if (account.count%20 == 0):
        date = get_last_datetime()
        feature, lable = Logisticdata(account.security, get_date_delta(date.strftime('%Y-%m-%d'),-40), get_date_delta(date.strftime('%Y-%m-%d'),-40))
        if (len(feature) != 0):
            feature = account.nm.fit_transform(feature)
            account.clf.fit(feature, lable) 
            buylist = listget(account.security, date, account)
            if len(buylist)>0:
                for stk in list(account.positions):
                    if stk not in buylist:
                        order_target(stk, 0)
                for stk in buylist:
                    order_value(stk, account.cash/len(buylist))

"""
    思考：对于机器学习算法我们应该不断地思考如何能够将其优化，这次逻辑回归在之前的随机森林的基础
    上增加了循环model fit的优化，但是我们仍然有很多东西可以优化比如说分类阀值（n天后的涨幅），调
    仓频率，此外对于机器学习模型有一个很重要的需要优化的点-参数我们在三个模型中都没有进行优化，而
    对于机器学习模型参数的优化主要是利用交叉验证进行循环取最优结果。之后的章节我们将对这一点进行
    讨论
"""               