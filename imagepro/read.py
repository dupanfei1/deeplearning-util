# -*- coding: utf-8 -*-
"""
Created on Sun May 27 11:38:41 2018

@author: root
"""

import numpy as np
import os
import shutil
import random

# Dict={}
# with open('train.txt','r') as f:
#     data = f.readlines()
#
#     for line in data:
#         b = line.split(' ')
#         # print b[0]
#         Dict[b[0]]=b[1][0:-1]
#
# for filename in os.listdir(r"train2"):
#     b = filename.split('.')[0]
#     b = filename
#     file_dir='train/'+str(Dict[b])+'/'
#     if not os.path.isdir(file_dir):
#         os.makedirs(file_dir)
#     shutil.copy('train2/'+filename,'train/'+str(Dict[b])+'/')
# print("train1 finish")
#


# for line in  f.readlines()[999:len(f.readlines())-1]:
#     print(line)


#copy data
for i in range(1,101):

    for filename in os.listdir(r'trainadd0606/' + str(i) + '/'):
        shutil.copy('trainadd0606/' + str(i) + '/' + filename, 'trainrename/')
print("train1 finish")


# with open('0531.csv', 'r') as f:
#     data = f.readlines()
#
#     for line in data:
#         b = line.split(' ')
#         Dict[b[0]] = b[1][0:-1]
#
# for filename in os.listdir(r"data/test"):
#     b = filename.split('.')[0]
#     b = filename
#     file_dir = 'test/' + str(Dict[b]) + '/'
#     if not os.path.isdir(file_dir):
#         os.makedirs(file_dir)
#     shutil.copy('data/test/' + filename, 'test/' + str(Dict[b]) + '/')
# print("train1 finish")