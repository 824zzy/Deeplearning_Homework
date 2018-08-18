#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 14:21
# @Author  : zzy824
# @File    : Shakespeare_writing_main.py

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from shakespeare_utils import *
import sys
import io


# todo：what the fuck is LambdaCallback?
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])

# Run this cell to try with different inputs without having to re-train the model
generate_output()