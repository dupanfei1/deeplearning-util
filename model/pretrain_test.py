# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:50:03 2017

@author: lab548
"""

import os, random
# os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import keras.models as models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Bidirectional
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import ZeroPadding2D
from keras.regularizers import *
from keras.layers.recurrent import LSTM, GRU
from keras.models import Model
from keras import optimizers
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
import h5py
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
from collections import OrderedDict
from scipy.fftpack import fft, ifft

# Xd = cPickle.load(open("/home/dupanfei/Desktop/new1113/modulate/data10a-6.dat", 'rb'))

X1 = cPickle.load(open("/home/dupanfei/Desktop/new1113/pretrain/bpsknew.dat",'rb'))
X2 = cPickle.load(open("/home/dupanfei/Desktop/new1113/pretrain/qpsknew.dat",'rb'))
X3 = cPickle.load(open("/home/dupanfei/Desktop/new1113/pretrain/16qamnew.dat",'rb'))
X4=  cPickle.load(open("/home/dupanfei/Desktop/new1113/pretrain/4fsknew.dat",'rb'))
X5= cPickle.load(open("/home/dupanfei/Desktop/new1113/pretrain/2fsknew.dat",'rb'))
X6 = cPickle.load(open("/home/dupanfei/Desktop/new1113/pretrain/8psknew.dat",'rb'))
X7 = cPickle.load(open("/home/dupanfei/Desktop/new1113/pretrain/64qamnew.dat",'rb'))

Xd=dict(X1.items()+X2.items()+X3.items()+X4.items()+X5.items()+X6.items()+X7.items())

snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))
X = np.vstack(X)

np.random.seed(2018)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

in_shp = list(X_train.shape[1:])
print X_train.shape, in_shp
classes = mods

NUM_CNN_OUTPUTS = 64
CNN_KERNEL = (1, 6)

NUM_LSTM_OUTPUTS = 64

l = in_shp[1]
# 组合神经网络
#
nb_epoch = 50  # number of epochs to train on
batch_size = 256  # training batch size

model = model_from_json(open('para.json').read())

model.compile(loss='categorical_crossentropy', optimizer='adamax',metrics=['accuracy'])
model.summary()
model.load_weights('para.h5')

score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print score

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, X_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] = conf[j, k] + 1
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
plot_confusion_matrix(confnorm, labels=classes)

acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)" % (snr))

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print "Overall Accuracy: ", cor / (cor + ncor)
    acc[snr] = 1.0 * cor / (cor + ncor)
dict1 = sorted(acc.values())

plt.figure()
plt.title('Training performance')
plt.plot(snrs, dict1)

# Plot accuracy curve
plt.plot(snrs, map(lambda x: acc[x], snrs))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("Classification Accuracy")
plt.show()