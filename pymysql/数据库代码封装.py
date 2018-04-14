#coding=utf-8   
#Version:python3.6.0
#tools:Pycharm 2017.3.2
__Date__ = '2018/2/17 19:15'
__Author__ = 'admin'
import pymysql
class Database(object):
    _config ={
        "host":"192.168.1.2",
        "user":"root",
        "password":"123123",
        "db":"db_test",
        "charset":"utf8"
    }
    def __init__(self):
        self.db = None
        self.cursor = None
    def head(self,sql,*args):
        try:
            self.db = pymysql.connect(**Database._config)
            self.cursor = self.db.cursor()
            self.cursor.execute(sql,*args)
            return self.cursor.fetchone()
        except Exception as error:
            print(error,error)
        finally:
            self.close()
    def get(self,sql,*args):
        try:
            self.db = pymysql.connect(**Database._config)
            self.cursor = self.db.cursor()
            self.cursor.execute(sql,*args)
            return self.cursor.fetchall()
        except Exception as error:
            print(error,error)
        finally:
            self.close()
    #增删改
    def execute(self,sql,*args):
        try:
            self.db = pymysql.connect(**Database._config)
            self.cursor = self.db.cursor()
            num = self.cursor.execute(sql,*args)
            self.db.commit()
            return num#sql语句执行之后影响行数
        except Exception as error:
            self.db.rollback()
            print(error,error)
        finally:
            self.close()
    def close(self):
        # 关闭数据库连接
        if self.cursor:
            self.cursor.close()
        if self.db:
            self.db.close()
if __name__ == "__main__":
    db = Database( )
    # print(db.execute("insert into t_student VALUES (%s,%s,%s,%s) ",(4,'ww',30,"北京")))
    # db.head()
    print(db.get("select * from t_student"))