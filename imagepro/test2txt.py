#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
"""
"""
Dir = "/home/lab548/Desktop/baidu/saler_torch_3/trainadd0606/"
images_names = [0]*100
f = open("train_add0606.txt", "w")
namelist=[]
for i in range(1,101):
    imagesDir = os.path.join(Dir + str(i))
    images_names[i-1] = os.listdir(imagesDir)
    for file in images_names[i-1]:
        filename = file + ' ' + str(i)
        if file not in namelist:
            f.writelines('%s\n' % filename)
            namelist.append(file)
    print 'the',i,'line has done'
f.close
