#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:51:07 2019

@author: fanzy
"""

from utils import loaddata
from utils import loss_history
from utils import plot_loss
import os
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

path='/home/fanzy/data/catdog/train/'
size=227
x_train,x_test,y_train,y_test=loaddata.catdogimg(path,size,0.3)

model=Sequential()






