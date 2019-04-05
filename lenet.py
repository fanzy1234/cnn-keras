#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:49:36 2019

@author: fanzy
"""
from utils import loaddata
from utils import loss_history
from utils import plot_loss
import os
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

path='/home/fanzy/data/catdog/train/'
size=28
x_train,x_test,y_train,y_test=loaddata.catdogimg(path,size,0.3)

model=Sequential()
model.add(Conv2D(filters=6,kernel_size=(5,5),padding='valid',
                 input_shape=(size,size,3),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='valid',activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120,activation='tanh'))
model.add(Dense(84,activation='tanh'))
model.add(Dense(2,activation='softmax'))

sgd=SGD(lr=0.05,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

history=loss_history.LossHistory()
model.fit(x_train,y_train,batch_size=200,epochs=20,
          validation_data=(x_test, y_test),
          verbose=1,shuffle=True,callbacks=[history])

#plot_model(model,to_file='lenet.png',show_shapes=True,show_layer_names=False)

score=model.evaluate(x_test,y_test)

plot_loss.plot(history.losses,history.accuracy,
               history.val_losses,history.val_accuracy)


