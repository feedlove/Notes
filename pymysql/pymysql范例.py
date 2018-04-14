#coding=utf-8   
#Version:python3.6.0
#tools:Pycharm 2017.3.2
__Date__ = '2018/2/17 15:25'
__Author__ = 'admin'
import numpy as np
import pandas as pd
import pymysql
import datetime
#创建数据库连接
# 打开数据库连接
db = pymysql.connect("192.168.1.2","root","123123","db_test",charset='utf8' )
## 使用cursor()方法获取操作游标
cursor = db.cursor()
# 使用 execute() 方法执行 SQL，如果表存在则删除
# cursor.execute("select * from 年度数据")
# print(cursor.fetchall())
# SQL 查询语句
#sql = "insert into t_student values (2,'w',25,'上海') "
sql = "insert into t_student values (%s,%s,%s,%s)"
try:
    #执行SQL语句
    cursor.execute(sql,args=(3,'w',25,'上海'))
    db.commit()
    #获取所有记录列表
    # results = cursor.fetchone()
    # print(results)

except Exception as error:
    db.rollback()
    print(error)
finally:
    # 关闭数据库连接
    if cursor:
        cursor.close()
    if db:
        db.close()
    # print("The log was generated in %s")%(datetime.datetime.now().strftime('%Y{y}%m{m}%d{d}'.format(y='年',m='月',d='日')))
