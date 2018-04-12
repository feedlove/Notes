# 随机森林基于多因子选股
"""
    策略思想：针对沪深300（000300.SH 沪深300）中所有股票，选出7个因子
    （zdzb：ZDZB筑底指标,market_cap:总市值 ,obv:OBV能量潮 ,pe_ttm:市
    盈率TTM ,boll:BOLL布林线 ,pb:市净率 ,kdj:KDJ随机指标)作为训练样本
    特征属性，选择一天（2015/01/05)的所有沪深300的股票的七个因子作为
    训练集，并进行最大最小标准化操作，对于类别值以20天为基准，20日后
    涨幅超过阀值（10%）为1，反之为-1，然后利用随机森林算法进行模型训
    练，然后每个回测日对当前一个交易日的沪深300所有股票的这七个因子先
    进行标准化操作然后再用随机森林模型预测，然后每日选出预测类别为1概
    率最大三只股票，若现在持仓的股票不在这三只股票中，则全部卖出，然
    后在等权买入这三只股票。
    
    初始资金：1000000
    回测频率：每天
    回测日期：2015-01-01——2017-04-01
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

#对时间操作的时间包
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

def randomclfdata(stklist, start, end):
    """获取随机森林的训练数据特征值以及相应的类别
    
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
        
        #获得特定日期的因子数据
        df = get_factor_data(stklist, traindate)
        
        #获得相应的类别，20日收益率大于10%为1小于10%为-1.（由于get_price函数得出的不是一个DataFrame，所以对于多只股票很难进行批量化操作，然而一个个循环股票代码速度十分缓慢因此利用lambda）
        df['label'] = df['factor_symbol'].map(lambda x: 1 if get_price([x], start_date = get_date_delta(traindate,1), end_date = get_date_delta(traindate,21), fields = ['open'])[x].open[-1]/get_price([x], start_date = get_date_delta(traindate,1), end_date = get_date_delta(traindate,21), fields = ['open'])[x].open[0] - 1 >= 0.1 else -1)
        
        df = df.set_index('factor_symbol')
        df = df.dropna()
    
    #若有多个日期的因子则将其集合起来
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
    
    #得到回测前一个交易日的特征值
    tdf = get_factors(q).set_index('factor_symbol').dropna()
    
    buy = []
    if len(tdf) > 0:
        feature_test = account.nm.transform(tdf.values)
        lable_test = account.clf.predict_proba(feature_test)[:,1]
        ans = pd.DataFrame(lable_test, index = tdf.index.values)
        #log.info(ans)
        #对股票得到类别为1的概率进行排序并且选择最大的三只作为买入股票
        buy = ans.sort_values(ans.columns.values[0]).index.values[-3:]
    
    return buy
        
#初始化账户       
def initialize(account):      
    
    #设置要交易的证券(沪深300)      
    account.security = get_index_stocks('000300.SH', '20150105')
    
    #设置沪深300为基准
    set_benchmark('000300.SH')
    
    #将随机森林模型作为全局变量初始化
    account.clf = RandomForestClassifier(n_estimators = 10, max_depth = 3, random_state=0)
    
    feature, lable = randomclfdata(account.security, '20150105', '20150105')
    
    #将标准化模型作为全局变量初始化
    account.nm = MinMaxScaler()
    
    feature = account.nm.fit_transform(feature)
    account.clf.fit(feature, lable) 
    
    #用来控制调仓频率
    account.count = -1
    
#设置买卖条件，每个交易频率（日/分钟/tick）调用一次   
def handle_data(account,data):
    account.count += 1
    
    #对n求余则每n天调一次仓
    if account.count%1 == 0:
    
        date = get_last_datetime()
        buylist = listget(account.security, date, account)
        if len(buylist)>0:
            apk = list(account.positions.keys())
            for stk in apk:
                if stk not in buylist:
                    order_target(stk, 0)
            for stk in buylist:
                
                #进行等比例买入
                order_value(stk, account.cash/len(buylist))

"""
    思考：尽管这次回测表现比第一章决策树择时要好，而且代码变成了模块化，标准化的代码方便日后的引用。但是我们仍旧要思考是否有能够改进的
    地方。比如说我们模型的fit只利用某一天的HS300的训练样本训练了一次，我们应该考虑是不是进行循环fit（既每个回测日用当前n天的数据当做
    训练集进行模型的fit）的话可能表现会更好，此外是不是我们每天一调仓变成n天一调仓会变得更好？让我们进入第三章机器学习之逻辑回归。

"""           