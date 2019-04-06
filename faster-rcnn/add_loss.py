#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:27:38 2019

@author: fanzy
"""
from keras.models import Model
import keras.layers as KL
import keras.backend as K
import numpy as np
from keras.utils.vis_utils import plot_model

x_train=np.random.normal(1,1,(100,784))

x_in = KL.Input(shape=(784,))
x = x_in
x = KL.Dense(100, activation='relu')(x)
x = KL.Dense(784, activation='sigmoid')(x)
def custom_loss1(y_true,y_pred):
    return K.mean(K.abs(y_true-y_pred))
loss1=KL.Lambda(lambda x:custom_loss1(*x),name='loss1')([x,x_in])

model = Model(x_in, [loss1])
model.get_layer('loss1').output
model.add_loss(loss1)
model.compile(optimizer='adam')
plot_model(model,to_file='model.png',show_shapes=True)
#
model.fit(x_train, None, epochs=5)

x_in = KL.Input(shape=(784,))
x = x_in
x = KL.Dense(100, activation='relu')(x)
x = KL.Dense(784, activation='sigmoid')(x)

model = Model(x_in, x)
loss = K.mean((x - x_in)**2)
model.add_loss(loss)
model.compile(optimizer='adam')
plot_model(model,to_file='model.png',show_shapes=True)
model.fit(x_train, None, epochs=5)