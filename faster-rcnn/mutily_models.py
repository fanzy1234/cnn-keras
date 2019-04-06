#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 08:39:40 2019

@author: fanzy
"""

from keras.models import Model
import keras.layers as KL
import keras.backend as K
import numpy as np
from keras.utils.vis_utils import plot_model


def custom_loss1(y_true,y_pred):
    return K.mean(K.abs(y_true-y_pred))
def custom_loss2(y_true,y_pred):
    return K.mean(K.abs(y_true-y_pred))

#三个输入向量
input1=KL.Input((32,32,3))
input2=KL.Input((4,))
input3=KL.Input((2,))

#对input1做操作得到temp1
temp1=KL.BatchNormalization(axis=1)(input1)
temp1=KL.Conv2D(16,(3,3),padding='same')(temp1)
temp1=KL.Activation('relu')(temp1)
temp1=KL.MaxPooling2D(2)(temp1)
temp1=KL.Flatten()(temp1)
temp1=KL.Dense(2)(temp1)
#对input2做操作得到temp2
temp2=KL.Dense(32)(input2)
temp2=KL.Dense(2)(temp2)

#temp1,temp2计算得到loss1 ,通过Lambda自定义层
#对temp1,input3计算得到loss2
loss1=KL.Lambda(lambda x:custom_loss1(*x),name='loss1')([temp1,temp2])
loss2=KL.Lambda(lambda x:custom_loss2(*x),name='loss2')([temp1,input3])
#将输入输出放进model中，建立网络
model=Model([input1,input2,input3],[loss1,loss2])
plot_model(model,to_file='model.png',show_shapes=True)#查看model 网络结构
#将自定义的loss层的结果取出作为model的loss
loss_layer1=model.get_layer('loss1').output
loss_layer2=model.get_layer('loss2').output
model.add_loss(loss_layer1)
model.add_loss(loss_layer2)

model.compile(optimizer='sgd',loss=[None,None])
#yield把函数变成一个生成器，逐块将数据载入，而不是一下子全部载入，减小显存占用
def data_gen(num):
    for i in range(num):
        yield [np.random.normal(1,1,(1,32,32,3)),
               np.random.normal(1,1,(1,4)),np.random.normal(1,1,(1,2))],[]
dataset=data_gen(10000)

model.fit_generator(dataset,steps_per_epoch=200,epochs=20)







