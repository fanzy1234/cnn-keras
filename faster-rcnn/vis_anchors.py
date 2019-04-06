# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#featuremap size16*16
size_x=4
size_y=4

rpn_stride=16#下采样比例，input与featuremap之间的尺寸比 inputsize=16*8
scale=[8,16,32]#一个featuremap点对应２*2，４*4，８*8三种面积，
ratio=[0.5,1,2]#每种面积对应３种不同的长宽尺寸


def anchor_gen(size_x,size_y,rpn_stride,scale,ratio):
#    featuremap_x=np.arange(size_x)#[0,1,2,...15]
#    featuremap_y=np.arange(size_y)
#    #featuremap的每一个点的坐标
#    featuremap_x,featuremap_y=np.meshgrid(featuremap_x,featuremap_y)#featuremap16*16个点的横纵坐标
    
    shift_x=np.arange(size_x)*rpn_stride#[0,8,16,...120]
    shift_y=np.arange(size_y)*rpn_stride
    #(input_xs,input_ys)featuremap上每一个点对应的原图上的中心点，也即anchor锚点
    shift_x,shift_y=np.meshgrid(shift_x,shift_y)#原图上anchor的16*16个锚点坐标
    #featuremap每个点对应的anchor缩放比例　
    scales,ratios=np.meshgrid(scale,ratio)
    scales,ratios=scales.flatten(),ratios.flatten()
    anchor_h=scales*np.sqrt(ratios)#anchor height
    anchor_w=scales/np.sqrt(ratios)#anchor width
    
    center_x,anchor_w=np.meshgrid(shift_x,anchor_w)#centex anchorw一一对应
    center_y,anchor_h=np.meshgrid(shift_y,anchor_h)
    
    anchor_center=np.stack([center_x,center_y],axis=2).reshape(-1,2)
    anchor_h_w=np.stack([anchor_h,anchor_w],axis=2).reshape(-1,2)
    
    anchors=np.concatenate([anchor_center-0.5*anchor_h_w,anchor_center+0.5*anchor_h_w],axis=1)

    return anchors

anchors=anchor_gen(size_x,size_y,rpn_stride,scale,ratio)

img=np.ones((128,128,3))
plt.imshow(img)
axs=plt.gca()
for i in range(anchors.shape[0]):
    anchor=anchors[i]
    rec=patches.Rectangle((anchor[0],anchor[1]),anchor[2]-anchor[0],anchor[3]-anchor[1],edgecolor='r',facecolor='none')
    axs.add_patch(rec)

#
























