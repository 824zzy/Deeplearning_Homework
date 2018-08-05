#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/4 22:10
# @Author  : zzy824
# @File    : optimization_methods.py

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from utils.opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from utils.opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from utils.testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'