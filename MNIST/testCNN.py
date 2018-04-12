# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:41:16 2018

@author: Administrator
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean',mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
		tf.summary.scalar('stddev',stddev)
		tf.summary.scalar('max',tf.reduce_max(var))
		tf.summary.scalar('min',tf.reduce_min(var))
		tf.summary.hisogram('histogram',var)


def weight_variable(shape,name):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1),name=name)

def bias_variable(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def CNN():
# =============================================================================
# =============================================================================
    mnist = input_data.read_data_sets('D:\python\MNIST', one_hot=True)
    batch_size = 100
    n_batch = mnist.train.num_examples//batch_size
      
  #   input
    with tf.name_scope('input'):
        x = tf.placeholder("float", shape=[None, 784])
        y_ = tf.placeholder("float", shape=[None, 10])
    with tf.name_scope('x_image'):
        x_image = tf.reshape(x, [-1,28,28,1])
# =============================================================================
#     #第一层卷积
# =============================================================================
    with tf.name_scope('Conv1'):
        with tf.name_scope('W_conv1'):
            W_conv1 = weight_variable([5, 5, 1, 32],name='W_conv1')
        with tf.name_scope('b_conv1'):
            b_conv1 = bias_variable([32],name='b_conv1')    
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope('pool'):
        h_pool1 = max_pool_2x2(h_conv1)
     
    #第二层卷积
    with tf.name_scope('Conv2'):
        with tf.name_scope('W_conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64],name='W_conv2')
        with tf.name_scope('b_conv2'):
            b_conv2 = bias_variable([64],name='b_conv2')
        with tf.name_scope('relu'):
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        with tf.name_scope('pool'):
            h_pool2 = max_pool_2x2(h_conv2)
    #密集连接层
    with tf.name_scope('fc1'):
        with tf.name_scope('W_fc1'):
            W_fc1 = weight_variable([7 * 7 * 64, 1024],name='W_fc1')
        with tf.name_scope('b_fc1'):
            b_fc1 = bias_variable([1024],name='b_fc1')
        with tf.name_scope('h_pool2_flat'):
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64],name='h_pool2_flat')
        with tf.name_scope('relu'):
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #Dropout
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder("float",name='keep_prob')
    with tf.name_scope('keep_prob'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name='h_fc1_drop')
    #输出层
    with tf.name_scope('fc2'):
        with tf.name_scope('W_fc2'):
            W_fc2 = weight_variable([1024, 10],name='W_fc2')
        with tf.name_scope('b_fc2'):
            b_fc2 = bias_variable([10],name='b_fc2')
        with tf.name_scope('softmax'):
            y=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    #训练和评估模型
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y),name='cross_entropy')
        tf.summary.scalar('cross_entropy',cross_entropy)
    with tf.name_scope('train_step'):
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
		
		
    with tf.name_scope('accuracy'):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            tf.summary.scalar('accuracy',accuracy)
			
	
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("D:/python/logs/train",sess.graph)
        test_writer = tf.summary.FileWriter("D:/python/logs/test",sess.graph)
        for i in range(1000):
            batch_train = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_train[0],y_:batch_train[1],keep_prob:0.7})
            summary = sess.run(merged,feed_dict={x:batch_train[0],y_:batch_train[1],keep_prob:1.0})
            train_writer.add_summary(summary,i)
            	  
            batch_test = mnist.test.next_batch(batch_size)
            summary = sess.run(merged,feed_dict={x:batch_test[0],y_:batch_test[1],keep_prob:1.0})
            test_writer.add_summary(summary,i)
            if i%100 == 0:
                train_accuracy = sess.run(accuracy,feed_dict={
                    x:batch_train[0], y_: batch_train[1], keep_prob: 1.0})
                test_accuracy = sess.run(accuracy,feed_dict={
                    x:batch_test[0], y_: batch_test[1], keep_prob: 1.0})		
                print('Iter'+str(i)+'Training Accuracy:'+str(train_accuracy)+'Testing Accuracy:'+str(test_accuracy))

CNN()