#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/3 15:19
# @Author  : zzy824
# @File    : Logistic_Regression_with_a_Neural_Network_mindset.py

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


""" data_set
- a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
- a test set of m_test images labeled as cat or non-cat
- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).
"""
# Loading the data(cat/non-cat)
"""
train_set_x_orig and test_set_x_orig are going to preprocess: (the number of examples, num_px, num_px, channel) like (209, 64, 64, 3)

train_set_y and test_set_y are labels: (class, the number of examples) like（1, 209）
"""
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Reshape the training and test examples
# todo: A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b ∗∗ c ∗∗ d, a) is to use: X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
"""
shape of train_set_x and test_set_x will be (12288, 209) and (12288, 50)
"""
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standardize our dataset
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


# helper functions
def sigmoid(z):
    """
    computer the sigmoid of z

    :param z: A schalar or numpy array of any size
    :return: s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    :param dim: size of the w vector we want
    :return:
    """
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    """
    implement the cost function and its gradient for the propagation

    :param w: weights, a numpy array of size(num_px * num_px * 3, 1)
    :param b: bias , a scalar
    :param X: data of size (num_px * num_px * 3, number of examples)
    :param Y: true label vector of size(1, number of examples)
    :return:
    """
    # m = num_px * num_px * 3, or input size of neural network
    m = float(X.shape[1])

    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)

    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # Backward propagation
    dw = 1 / m * np.dot(X, (A - Y).T)

    assert(dw.shape == w.shape)
    db = 1 / m * np.sum(A - Y)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """

    :param w: weights, a numpy array of size(num_px * num_px * 3, 1)
    :param b: bias , a scalar
    :param X: data of size (num_px * num_px * 3, number of examples)
    :param Y: true label vector of size(1, number of examples)
    :param num_iterations: numbers of iterations of the optimization loop
    :param learning_rate:
    :param print_cost: true to print the loss every 100 steps
    :return:
    """
    costs = []

    for i in range(num_iterations):
        # calculate grads and costs
        grads, cost = propagate(w, b, X, Y)
        # retrive derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print "Cost after iteration %i: %f" % (i, cost)

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


def predict(w, b, X):
    """

    :param w: weights, a numpy array of size(num_px * num_px * 3, 1)
    :param b: bias , a scalar
    :param X: data of size (num_px * num_px * 3, number of examples)
    :return:
    """
    # m equals to num_px * num_px * 3
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    # w = w.reshape(X.shape[0], 1)

    # predicted probability of X, size of (number of examples , 1)
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    assert(Y_prediction.shape == (1, m))

    return Y_prediction


# merge all functions to model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # retrieve parameters
    w = parameters["w"]
    b = parameters["b"]

    # predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


# run the model one time
# d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# show plot learning curve with costs
# costs = np.squeeze(d["costs"])
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iteration per hundreds')
# plt.title('learning rate =' + str(d["learning_rate"]))
# plt.show()

# choice of learning rate
learning_rate = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rate:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i, print_cost=False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rate:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

# todo: plt with legend!
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()