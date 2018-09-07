# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 10:03:31 2018

@author: raman
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir
import custom_functions as cf

imgs = []
objpoints = []
imgpoints = []

#get images...
files = listdir()
for index, fname in enumerate(files):
    if fname.find('.bmp') > -1:
#        fname = '{} Filter.bmp'.format('0'+str(i+1) if (i+1) < 10 else str(i+1))
        print('processesing', fname)
        img = cv2.imread(fname)
        foundCircles, cur_imgpoints, cur_objpoints = cf.getImageAndObjectPoints(img)
        
        if index == 0:
#            plt.subplot(221)
            plt.imshow(foundCircles)
        
        imgpoints.append(cur_imgpoints)
        objpoints.append(cur_objpoints)
        imgs.append(foundCircles)

