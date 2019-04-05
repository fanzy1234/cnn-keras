#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:04:09 2019
use keras model callbaacks to record train info
@author: fanzy
"""
import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy=[]
        self.val_losses=[]
        self.val_accuracy=[]

#    def on_batch_end(self, batch, logs={}):
#        self.losses.append(logs.get('loss')) 
#        self.accuracy.append(logs.get('acc')) 
#        self.val_losses.append(logs.get('val_loss')) 
#        self.val_accuracy.append(logs.get('val_acc'))
        
    def on_epoch_end(self, epoch, logs={}): 
        self.losses.append(logs.get('loss')) 
        self.accuracy.append(logs.get('acc')) 
        self.val_losses.append(logs.get('val_loss')) 
        self.val_accuracy.append(logs.get('val_acc'))

    
    
    
    
    
    