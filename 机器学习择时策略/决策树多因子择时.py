# 决策树多因子择时
"""
    策略思想：对一只股票（300033.SZ 同花顺），筛选出了7个因子（zdzb：
    ZDZB筑底指标,market_cap:总市值 ,obv:OBV能量潮 ,pe_ttm:市盈率TTM ,
    boll:BOLL布林线 ,pb:市净率 ,kdj:KDJ随机指标)作为训练样本特征属性，将
    2010/01/01-2014/12/30作为训练样本的数据的时间范围，并且利用按照训练
    样本对应的收益率对训练样本进行排序，只选取收益率前30%和后30%的数据作
    为真正的训练样本来去噪声，然后对因子进行标准化操作，对于类别值第二天
    涨为1，跌为-1，然后利用决策树算法进行模型训练，最后进行每日调仓，每个
    回测日对当前一个交易日的这七个因子先进行标准化操作然后再用决策树模型
    预测，预测结果为1就买入，预测结果为-1而且有持仓则全部卖出。
    
    初始资金：100000
    回测频率：每天
    回测日期：2015-04-01——2017-04-01
"""
import numpy as np
import pandas as pd

# 导入决策树的包
from sklearn.tree import DecisionTreeClassifier          
# 导入因子min-max标准化的包
from sklearn.preprocessing import MinMaxScaler

# 训练样本开始时间
start = '20100101' 
# 训练样本结束时间
end = '20141230'

# 得到要处理的股票的信息
s = ['300033.SZ']
x = get_price(s, start, end, '1d', ['close', 'open'], True)

# 选取因子
trade_days = get_trade_days(start, end).strftime('%Y-%m-%d')
q = query(
    factor.date,
    factor.zdzb,
    factor.market_cap,
    factor.obv,
    factor.pe_ttm,
    factor.boll,
    factor.pb,
    factor.kdj
).filter(
    factor.symbol == s[0],
    factor.date.in_(trade_days)
)
df = get_factors(q).set_index('factor_date')
df = df.dropna()
x = pd.DataFrame(x[s[0]])

# 计算收益率，由于在每日结盘后才能获得关于当天的因子，因此买入最早是在第二天的开盘，而卖出则最早则在买入的后一天的开盘
x['return'] = np.log(x['open']).diff(1).shift(-2)

# 将因子数据以及股票收益率数据融合成一个DataFrame
x = x.merge(df, how = 'inner', left_index= True, right_index= True)
x = x.dropna()

# 二值化去噪声，只选取收益率中的前30%以及后30%
x = x.sort_values('return')
x = x.iloc[:int(len(x)*0.3), :].append(x.iloc[int(len(x)*0.7):,:])

# m为类别判断阀值，收益率大于m为1，小于m为-1
m = max(x['return'].mean(),0)

for i in range(len(x)):
    if x['return'].iloc[i] >= m :
        x['return'].iloc[i] = 1
    else:
        x['return'].iloc[i] = -1

#对特征值（因子）进行标准化操作
ds = MinMaxScaler()
feature = ds.fit_transform(x.iloc[:,3:].values)
lable = x['return'].T.values 

#利用决策树进行训练
clf = DecisionTreeClassifier(random_state=0)
clf.fit(feature, lable)

#初始化账户       
def initialize(account):      
    #设置要交易的证券(000001.SZ 平安银行)      
    account.security = '000001.SZ' 
    #设定基准
    set_benchmark('000001.SZ')

#设置买卖条件，每个交易频率（日/分钟/tick）调用一次   
def handle_data(account,data):
    #获取回测昨日的因子（特征值）数据
    date = get_last_datetime().strftime("%Y-%m-%d")
    q = query(
        factor.date,
        factor.zdzb,
        factor.market_cap,
        factor.obv,
        factor.pe_ttm,
        factor.boll,
        factor.pb,
        factor.kdj
    ).filter(
        factor.symbol == account.security,
        factor.date == date
    )
    df = get_factors(q).set_index('factor_date')   
    df = df.dropna()
    
    if len(df) > 0:
        #标准化回测日因子
        feature_test = ds.transform(df.values)
        lable_test = clf.predict(feature_test)
        #预测值为1买入，-1且有仓位卖出
        if (lable_test == 1):
            order_value(account.security, account.cash)
        else:
            if (account.positions_value>0):
                order_target(account.security, 0)
                
"""
    思考：尽管策略跑赢了基准，但是策略表现并不好，除此之外，决策树由于过度分支可能会造成过拟合的
    现象，另外，由于此代码适用于初学者代码写的不够规范，不够模块化。但是最重要的是我们应该怎样来
    优化我们的策略以及代码。一个很容易得到的想法就是我们可以利用多颗策略树来进行判断，这样既能够
    很好的解决掉过拟合的影响也能够得到更加准确的结果。我们进入第二章机器学习之随机森林。
"""