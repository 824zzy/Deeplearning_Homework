#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 18:47
# @Author  : zzy824
# @File    : tensorflow_tutorial.py

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from utils.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)

# loss function in tensorflow
y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')

loss = tf.Variable((y-y_hat)**2, name='loss')
# todo: https://blog.csdn.net/u012436149/article/details/78291545
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print "loss: " + str(session.run(loss))
    print "----------"
    session.close()

# session
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a, b)
print(c)
sess = tf.Session()
print(sess.run(c))

# placeholder
x = tf.placeholder(tf.int64, name='x')
print(sess.run(2 * x, feed_dict={x: 3}))
print("----------")
sess.close()

# linear function
def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """

    np.random.seed(1)

    X = tf.constant(np.random.randn(3, 1), name="X")
    W = tf.constant(np.random.randn(4, 3), name="W")
    b = tf.constant(np.random.randn(4, 1), name="W")
    Y = tf.add(tf.matmul(W, X), b)

    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
    sess = tf.Session()
    result = sess.run(Y)

    # close the session
    sess.close()

    return result


print("result = " + str(linear_function()))

