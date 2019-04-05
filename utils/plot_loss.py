#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:04:09 2019
use keras model callbaacks to record train info
@author: fanzy
"""
import matplotlib.pyplot as plt

def plot(loss,accuracy,val_loss,val_acc):
    iters = range(len(loss))
    #创建一个图
    plt.figure()
    # acc
    plt.plot(iters, accuracy, 'r', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
    # loss
    plt.plot(iters, loss, 'g', label='train loss')
    plt.plot(iters, val_acc, 'b', label='val acc')
    # val_loss
    plt.plot(iters, val_loss, 'k', label='val loss')
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel('acc and loss')#给x，y轴加注释
    plt.legend(loc="upper right")
    plt.show()

    
    
    
    
    
    