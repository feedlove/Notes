# SVM基于财务因子选股策略

"""
    策略思想：以季度为单位，选取一些财务因子作为训练样本，对于类别值，
            如果未来一个季度涨幅超过5%标记为1，反之为-1，然后用支持向
            量机算法进行训练，预测结果为1且未持仓则买入，预测结果为-1
            且已持仓则卖出。
    
    特征因子选取：选取2014-06-30（2014年半年报）时刻，沪深300成分股的
            市盈率、市净率、总市值、流通市值、净资产收益率roe加权、总资
            产报酬率roa、销售净利率、净利润／营业总收入、息税前利润／营
            业总收入、基本每股收益同比增长率、每股经营活动产生的现金流量
            净额同比增长率、营业总收入同比增长率、净利润同比增长率、经营
            活动产生的现金流量净额同比增长率、应收账款周转天数、应付账款
            周转天数、存货周转率、应收账款周转率、应付账款周转率、流动比
            率、速动比率共21个财务因子。

    数据标准化：数据标准化方法有很多，本文采用高斯预处理方法，即每个特征
            因子减去它对应的均值再除以它的标准差。

    参数优化：本文使用交叉验证的方法对惩罚参数与径向基函数（高斯核函数）
            参数Sigma进行优化筛选。交叉验证方法具体为选取的训练样本集分为
            K份，依次选取其中的K-1份作为训练集，剩下一份作为验证集，通过
            算法在验证集上的表现进行打分，再对打分进行取平均作为此参数对
            应的分数，依次选择不同的参数进行打分，确定最终的参数。
 
 初始资金：10000000
 回测频率：每天
 回测日期：2016-03-01 — 2017-03-01
    
"""

import pandas as pd 
import numpy as np 
from sklearn import svm 
from sklearn import cross_validation  
from sklearn import preprocessing
import datetime


def get_factor_data(stock_list, start_data):
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
    q = query(
        # 因子名称参考网页： 'http://quant.10jqka.com.cn/platform/html/help-api.html?t=data#222/0'
        factor.symbol,
        factor.pe,
        factor.pb,
        factor.market_cap,
        factor.capitalization,
        factor.weighted_roe,
        factor.roa,
        factor.net_profit_margin_on_sales,
        factor.net_profit_div_income,
        factor.before_tax_profit_div_income,
        factor.basic_pey_ear_growth_ratio,
        factor.net_cashflow_psg_rowth_ratio,
        factor.overall_income_growth_ratio,
        factor.net_profit_growth_ratio,
        factor.net_cashflow_from_opt_act_growth_ratio,
        factor.days_sales_outstanding,
        factor.days_payable_outstanding,
        factor.inventory_turnover_ratio,
        factor.turnover_ratio_of_receivable,
        factor.turnover_ratio_of_account_payable,
        factor.current_ratio,
        factor.quick_ratio
    ).filter(
        factor.symbol.in_(stock_list),
        factor.date == start_data
    )
    ans = get_factors(q)
    return ans
    
    
def create_data(start_data, end_data, stock_list):
    """ 数据集的生成 
    参数
    ----------
    start_data ： '%Y-%m-%d'形式字符串
        训练集选取开始日期
  
    end_data ： '%Y-%m-%d'形式字符串
        训练集选取结束日期
  
    stock_list : list类型
        股票列表
  
    返回值
    ----------
    df.values ： numpy ndarray结构
        训练样本特征集
 
    label ： list类型
        类别
    """
    # 调用get_factor_data函数获取特征因子数据
    df = get_factor_data(stock_list, start_data)
    #在df数据集上生成label（类别）列，涨幅超过5%标为1，反之为-1，注意lambda函数的应用
    df['label'] = df['factor_symbol'].map(lambda x: 1 if get_price([x], start_date = start_data, end_date = end_data, fields = ['open'])[x].open[-1]/get_price([x], start_date = start_data, end_date = end_data, fields = ['open'])[x].open[0] - 1 >= 0.05 else -1)
    # NaN值处理
    df = df.dropna()
    df.index = df['factor_symbol']
    label = list(df['label'])
    del(df['factor_symbol'], df['label'])
    return df.values, label

 
def optimize_C(X_train, X_test, y_train, y_test):
    """ 惩罚参数C优化
     参数
     ----------
     X_train ：numpy ndarray结构
        样本训练集特征因子数据
     
     X_test ：numpy ndarray结构
        样本验证集特征因子数据
     
     y_train ：numpy ndarray结构
        样本训练集类别数据
      
     y_test ：numpy ndarray结构
        样本验证集类别数据
     
     返回值
    ----------
     ans : float类型
        最优惩罚参数C
    """
    C_list = []
    score_list = []
    for i in range(100, 200, 1):  
        clf = svm.SVC(C = i)   
        #对训练数据进行训练
        clf.fit(X_train, y_train)  
        C_list.append(i)  
        #对测试集数据进行打分
        score_list.append(clf.score(X_test, y_test)) 
    ans = np.array(C_list)[np.array(score_list) == max(score_list)][0]
    return ans

 
def optimize_Sigma(X_train, X_test, y_train, y_test, C):
    """ 径向基函数参数Sigma优化
     参数
     ----------
     X_train ：numpy ndarray结构
        样本训练集特征因子数据
     
     X_test ：numpy ndarray结构
        样本验证集特征因子数据
     
     y_train ：numpy ndarray结构
        样本训练集类别数据
      
     y_test ：numpy ndarray结构
        样本验证集类别数据
     
     C : float类型
        最优惩罚参数
      
     返回值
    ----------
     ans : float类型
        最优径向基函数参数Sigma
    """
    Sigma_list = []  
    score_list = []  
    for i in range(100, 300, 1):  
        i=i/200
        clf = svm.SVC(C = C, gamma = i)
        clf.fit(X_train, y_train)  
        Sigma_list.append(i)  
        score_list.append(clf.score(X_test, y_test)) 
    ans = np.array(Sigma_list)[np.array(score_list) == max(score_list)][0]
    return ans
    
    
def initialize(account):
    # 基准日期选取2014-06-30，回测对象：沪深300成分股
    account.security = get_index_stocks(symbol = '000300.SH', date = '2014-06-30')
    # 获取训练集及类别
    df, label = create_data(start_data = '2014-06-30', end_data = '2014-09-30', stock_list = account.security)
    # 以下两行为数据标准化
    account.scaler = preprocessing.StandardScaler().fit(df)  
    df_scaler = account.scaler.transform(df) 
    # 以下三行为交叉验证获取最优参数值
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(df_scaler, label, test_size = 0.2, random_state = 0)
    C = optimize_C(X_train, X_test, y_train, y_test)
    Sigma = optimize_Sigma(X_train, X_test, y_train, y_test, C)
    # 优化参数之后，使用SVM算法进行训练
    account.clf = svm.SVC(C = C, gamma = Sigma)
    account.clf.fit(df_scaler, np.array(label))
    # 回测计数指标
    account.count = 0
    
    
def handle_data(account, data):
    # 设置调仓周期20天
    if account.count % 20 == 0:
        for stock in account.security:
            #以下二行为获取上一个交易日的因子数据
            yesterday_date = get_last_datetime()
            factor_data = get_factor_data([stock], yesterday_date.strftime('%Y-%m-%d'))
            del(factor_data['factor_symbol'])
            #以下为对获取的因子数据进行预测买卖且获取的测试集因子数据中可能有NaN值，避免出错加上异常处理
            try:
                #如果预测结果为1且未持仓则买入
                if account.clf.predict(account.scaler.transform(factor_data.values))[0] == 1 and stock not in account.positions.keys():
                    log.info('buying %s' %  stock)
                    order_percent(stock, 0.02)
                #如果预测结果为-1且已持仓则清仓
                if account.clf.predict(account.scaler.transform(factor_data.values))[0] == -1 and stock in account.positions.keys():
                    log.info('selling %s' %  stock)
                    order_target(stock, 0) 
            except Exception as e:
                pass
    # 回测计数指标加1
    account.count += 1