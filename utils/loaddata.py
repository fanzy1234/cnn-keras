#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:58:53 2019
load images
@author: fanzy

"""
import numpy as np
import os
import cv2
import random
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

def catdogimg(path,size,testsize):
    listImg=os.listdir(path)
    random.seed(40)
    random.shuffle(listImg)
    aryImgs=[]#[nums,H,W,channels]
    labels=[]
    for img in listImg:
        aryImg=cv2.imread(path+img)
        aryImg=cv2.resize(aryImg,(size,size))
        aryImg=np.asarray(aryImg,'f')#===
        aryImgs.append(aryImg)
    aryImgs=np.array(aryImgs,dtype='float')/255.0
    labels=[1 if x.split('.')[0]=='cat' else 0 for x in listImg]
    aryLabels= np_utils.to_categorical(labels,2)

    trainX,testX,trainY,testY=train_test_split(
                                aryImgs,aryLabels,test_size=testsize,random_state=40)
    return (trainX,testX,trainY,testY)


if __name__ == '__main__':
    path='/home/fanzy/data/catdog/train/'
    size=32
    x_train,x_test,y_train,y_test=catdogimg(path,size,0.3)












