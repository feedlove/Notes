# FM在特征组合中的应用

## 特征组合

|       | x1年龄 | x2北京 | x3上海 | x4深圳 | x5男 | x6女 |
| ----- | ------ | ------ | ------ | ------ | ---- | ---- |
| 用户1 | 23     | 1      | 0      | 0      | 1    | 0    |
| 用户2 | 31     | 0      | 0      | 1      | 0    | 1    |

如上例特征X有6个维度，年龄是连续值，城市和性别用one-hot表示，假设我们用最简单的线性拟合来预测y值。

​						$$y = w_0+\sum_{i=1}^{n}w_ix_i$$

实际中“北京的男性用户”、“上海的女性用户”这种组合特征可能是有用的，即$x_i,x_j$($x_i,x_j$都是onehot特征)，同时为1时可能是一个很有用的特征，这种组合特征是$x_i和x_j$的线性组合所无法表示的。这样一来乘积$x_ix_j$就成一个新的特征。为了不错过任何一个这种可能有用的组合特征，我们穷举所有的i,j组合，把$x_ix_j$,$1\le i \le n,i<j\le n$都加到特征里面去，即使其中某些$x_i$不是one-hot特征或者某些$x_ix_j$不是有用的特征，都没关系，经过大量样本的训练，模型会把那些无用的特征的系数训练为0。

​			$$y = w_o+\sum_{i=1}^{n}w_ix_i+\sum_{i}^{n}\sum_{j=i+1}^{n}w_{ij}x_ix_j$$

这只是组合了2个特征，同样道理我们组合任意三个特征、四个特征，随着阶数的提高，样本会显得非常稀疏，而且额外引入的参数呈指数增长。

## Factorization Machines 

由于二次项系数$w_{ij}$,我们额外引入$\frac{n^2}{2}]$个参数需要训练。有没有什么办法可以减少参数？再来观察二次项系数矩阵$W_{n*n}$，它是对称的方阵$w_{ij}=w_{ji}$，同时它是稀疏的，因为绝大部分的组合特征都是无用的，所以其系数应该为0。可以对$W_{n*n}$进行矩阵分解$W_{n*n}=V_{n*k}V_{n*k}^{T}$,即$w_{i,j}=<v_i,v_j>。其中$$k<<n$本来需要训练的n×n个参数，现在只需要训练n×k个。

![52500534584](C:\Users\ADMINI~1\AppData\Local\Temp\1525005345846.png)

![52500536842](C:\Users\ADMINI~1\AppData\Local\Temp\1525005368423.png)

![52500538065](C:\Users\ADMINI~1\AppData\Local\Temp\1525005380657.png)

![52500538869](C:\Users\ADMINI~1\AppData\Local\Temp\1525005388697.png)

![52500539590](C:\Users\ADMINI~1\AppData\Local\Temp\1525005395901.png)

## 代码实现 

实际当中我们很少会直接用线性拟合来做预测，通常会再套一层sigmoid函数。

尽量避免使用for循环，尽量使用numpy的矩阵运算，因为numpy的矩阵运算做了并行处理。　

~~~python
# coding=utf-8
__author__ = 'orisun'

import numpy as np
np.random.seed(0)
import random


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    sigmoid函数对z求一阶偏导
    :param z:
    :return:
    """
    return sigmoid(z) * (1 - sigmoid(z))


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """
        平方误差损失函数
        :param a: 预测值
        :param y: 真实值
        :return:
        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """
        损失函数对z求偏导
        :param z: x的线性函数
        :param a:
        :param y:
        :return:
        """
        return (a - y) * sigmoid_prime(z)


class FM(object):
    def __init__(self, train, valid, k, eta, maxecho, r2, cost=QuadraticCost):
        """
        构造函数
        :param train: 训练数据
        :param valid: 验证数据
        :param k: 矩阵V的第2维
        :param eta: 固定学习率
        :param maxecho: 最多迭代次数
        :param r2: R2小于该值后可停止迭代
        :param cost: 损失函数
        """
        self.train_x = train[:, :-1]
        self.train_y = train[:, -1:]
        self.valid_x = valid[:, :-1]
        self.valid_y = valid[:, -1:]
        self.var_y = np.var(self.valid_y)  # y的方差，在每轮迭代后计算R2时要用到
        self.k = k
        self.eta = float(eta)
        self.maxecho = maxecho
        self.r2 = r2
        self.cost = cost
        # 用正态分布随机初始化参数W和V
        self.w0 = np.random.randn()
        self.w = np.random.randn(1, self.train_x.shape[1])
        self.v = np.random.randn(self.train_x.shape[1], self.k)

    def shuffle_data(self):
        """
        每轮训练之前都随机打乱样本顺序
        :return:
        """
        ids = range(len(self.train_x))
        random.shuffle(ids)
        self.train_x = self.train_x[ids]
        self.train_y = self.train_y[ids]

    def predict(self, x):
        """
        根据x求y
        :param x:
        :return:
        """
        z = self.w0 + np.dot(self.w, x.T).T + np.longlong(
            np.sum((np.dot(x, self.v) ** 2 - np.dot(x ** 2, self.v ** 2)),
                   axis=1).reshape(len(x), 1)) / 2.0

        return z, sigmoid(z)

    def evaluate(self):
        """
        在验证集上计算R2
        :return:
        """
        _, y_hat = self.predict(self.valid_x)
        mse = np.sum((y_hat - self.valid_y) ** 2) / len(self.valid_y)
        r2 = 1.0 - mse / self.var_y
        print "r2={}".format(r2)
        return r2

    def update_mini_batch(self, x, y, eta):
        """
        平方误差作为损失函数，梯度下降法更新参数
        :param x:
        :param y:
        :param eta: 学习率
        :return:
        """
        batch = len(x)
        step = eta / batch
        z, y_hat = self.predict(x)
        y_diff = self.cost.delta(z, y_hat, y)
        self.w0 -= step * np.sum(y_diff)
        self.w -= step * np.dot(y_diff.T, x)
        delta_v = np.zeros(self.v.shape)
        for i in xrange(batch):
            xi = x[i:i + 1, :]  # mini_batch中的第i个样本。为保持shape不变，注意这里不能用x[i]
            delta_v += (np.outer(xi, np.dot(xi, self.v)) - xi.T ** 2 * self.v) * (y_diff[i])
        self.v -= step * delta_v

    def train(self, mini_batch=100):
        """
        采用批量梯度下降法训练模型
        :param mini_batch:
        :return:
        """
        for itr in xrange(self.maxecho):
            print "iteration={}".format(itr)
            self.shuffle_data()
            n = len(self.train_x)
            for b in xrange(0, n, mini_batch):
                x = self.train_x[b:b + mini_batch]
                y = self.train_y[b:b + mini_batch]
                learn_rate = np.exp(-itr) * self.eta  # 学习率指数递减
                self.update_mini_batch(x, y, learn_rate)

            if self.evaluate() > self.r2:
                break


def fake_data(sample, dim, k):
    """
    构造假数据
    :param sample:
    :param dim:
    :param k:
    :return:
    """
    w0 = np.random.randn()
    w = np.random.randn(1, dim)
    v = np.random.randn(dim, k)
    x = np.random.randn(sample, dim)
    z = w0 + np.dot(w, x.T).T + np.longlong(
        np.sum((np.dot(x, v) ** 2 - np.dot(x ** 2, v ** 2)),
               axis=1).reshape(len(x), 1)) / 2.0
    y = sigmoid(z)
    data = np.concatenate((x, y), axis=1)
    return z, data


if __name__ == "__main__":
    dim = 9  # 特征的维度
    k = dim / 3
    sample = 100
    z, data = fake_data(sample, dim, k)

    train_size = int(0.7 * sample)
    valid_size = int(0.2 * sample)
    train = data[:train_size]  # 训练集
    valid = data[train_size:train_size + valid_size]  # 验证集
    test = data[train_size + valid_size:]  # 测试集
    test_z = z[train_size + valid_size:]

    eta = 0.01  # 初始学习率
    maxecho = 200
    r2 = 0.9  # 拟合系数r2的最小值
    fm = FM(train, valid, k, eta, maxecho, r2)
    fm.train(mini_batch=50)

    test_x = test[:, :-1]
    test_y = test[:, -1:]
    print 'z=', test_z
    print "y=", test_y
    z_hat, y_hat = fm.predict(test_x)
    print "z_hat=", z_hat
    print "y_hat=", y_hat
~~~

