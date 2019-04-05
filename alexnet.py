#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:07:11 2019

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
# -*- coding: utf-8 -*-
#1st layer
model.add(Conv2D(filters=96,kernel_size=(11,11),padding='valid',
                 input_shape=(size,size,3),activation='tanh',strides=(4,4)))
model.add(MaxPooling2D(kernel_size=(3,3),strides=2))
model.add(BatchNormalization())
#2nd layer
model.add(Conv2D(filters=256,kernel_size=(5,5),padding='same',
                 activation='tanh',strides=(1,1)))
model.add(MaxPooling2D(kernel_size=(3,3),strides=(2,2)))
model.add(BatchNormalization())

#3id layer
model.add(Conv2D(filters=384,kernel_size=(3,3),padding='same',
                 activation='tanh',strides=(1,1)))
#4th layer
model.add(Conv2D(filters=384,kernel_size=(3,3),padding='same',
                 activation='tanh',strides=(1,1)))
#5th layer
model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',
                 activation='tanh',strides=(1,1)))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(Flatten)

#6th layer
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
#7th layer
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
#output
model.add(Dense(2),acticvation='softmax')

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

history=loss_history.LossHistory()
model.fit(x_train,y_train,batch_size=200,epochs=20,
          validation_data=(x_test, y_test),
          verbose=1,shuffle=True,callbacks=[history])

score=model.evaluate(x_test,y_test)

plot_loss.plot(history.losses,history.accuracy,
               history.val_losses,history.val_accuracy)