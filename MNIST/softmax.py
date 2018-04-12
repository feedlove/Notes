# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:24:49 2018

@author: Administrator
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
def load_mnist()：
    mnist = input_data.read_data_sets('D:\python\MNIST', one_hot=True)
    #读数据
    #训练集
    X_train = mnist.train.images#shape=(55000, 784)
    y_lables = mnist.train.labels#(55000, 10)
    #测试集
    X_test = mnist.test.images#(10000, 784)
    y_test = mnist.test.labels#(10000, 10)
    
    return X_train,y_lables,X_test,y_test
'''


def softmax():
    #读数据
    mnist = input_data.read_data_sets('D:\python\MNIST', one_hot=True)
    #参数初始化
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    #softmax
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    
    #损失函数
    y_ = tf.placeholder('float',[None,10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    
    #训练
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    #initialize
    init = tf.initialize_all_variables()
    
    sess = tf.Session()
    sess.run(init)
    
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
    #评估我们的模型
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print('the accuracy of  prediction is:' ,sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


softmax()   
        
        