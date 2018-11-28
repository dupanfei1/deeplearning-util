# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:38:17 2018

@author: dupanfei
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
from keras.utils import np_utils
import keras.models as models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import ZeroPadding2D
from keras.regularizers import *
from keras import optimizers
import seaborn as sns
import cPickle, random, sys, keras
import h5py
from keras.models import model_from_json
from keras.optimizers import SGD
# from keras.utils.vis_utils import plot_model
# Generate dummy data

# data1 = spio.loadmat('/home/dupanfei/Desktop/code0709/stft/1_1tai_stft/1_device1_huayin_1peak_stft.mat')
# data2 = spio.loadmat('/home/dupanfei/Desktop/code0709/stft/1_2tai_stft/1_device2_huayin_1peak_stft.mat')

# data1 = spio.loadmat('1_2tai_stft/device2_huayin_stft.mat')
# data2 = spio.loadmat('1_2tai_stft/device2_kongzhi_stft.mat')
# data3 = spio.loadmat('1_2tai_stft/device2_shuju_stft.mat')
# data4 = spio.loadmat('2_device5_huayin_stft.mat')

data1 = spio.loadmat('/home/dupanfei/Desktop/code0709/stft/data/1_1tai_stft/device1_1_huayin_stft.mat')
data2 = spio.loadmat('/home/dupanfei/Desktop/code0709/stft/data/1_2tai_stft/device1_2_huayin_stft.mat')
# data3 = spio.loadmat('/home/dupanfei/Desktop/new1113/stftdata/new-11005-point-ding-mi-hua_stft2.mat')
# data4 = spio.loadmat('/home/dupanfei/Desktop/new1113/stftdata/new-11006-point-ding-mi-hua_stft2.mat')
#
# data5 = spio.loadmat('/home/dupanfei/Desktop/new1113/stftdata/new-11007-point-ding-mi-hua_stft2.mat')
# data6 = spio.loadmat('/home/dupanfei/Desktop/new1113/stftdata/new-11008-point-ding-mi-hua_stft2.mat')
# data7 = spio.loadmat('/home/dupanfei/Desktop/new1113/stftdata/new-11009-point-ding-mi-hua_stft2.mat')
# data8 = spio.loadmat('/home/dupanfei/Desktop/new1113/stftdata/new-11010-point-ding-mi-hua_stft2.mat')

def readmat(data1,data2):

    data11 = data1['S_mol']
    data22 = data2['S_mol']
    # data11= np.transpose(data11, (2,0,1))
    # data22= np.transpose(data22, (2,0,1))

    l11 = data11.shape[2]
    l2 = data11.shape[1]#89
    l3 = data11.shape[0]#64
    print l11,l2,l3
    dev1 = np.zeros([l11, l3, l2], dtype=np.floating)
    for i in range(l11):
        dev1[i, :, :] = data11[:, :, i]


    l22 = data22.shape[2]
    dev2 = np.zeros([l22, l3, l2], dtype=np.floating)
    for i in range(l22):
        dev2[i, :, :] = data22[:, :, i]
    return dev1, dev2

dev1, dev2 = readmat(data1,data2)#读取mat文件
# dev3, dev4 = readmat(data3,data4)
# dev5, dev6 = readmat(data5,data6)
# dev7, dev8 = readmat(data7,data8)

Xd={}
Xd[('device10_tiao_shu',20)] = dev1
Xd[('device10_tiao_hua',20)] = dev2
# Xd[('device05',20)] = dev3
# Xd[('device06',20)] = dev4
#
# Xd[('device07',20)] = dev5
# Xd[('device08',20)] = dev6
# Xd[('device09',20)] = dev7
# Xd[('device10',20)] = dev8


snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):
            lbl.append((mod,snr))
X = np.vstack(X)
np.random.seed(2018)
n_examples = X.shape[0]
n_train = n_examples * 0.6
train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

# Y_train = np.array(map(lambda x: mods.index(lbl[x][0]), train_idx))
# Y_test = np.array(map(lambda x: mods.index(lbl[x][0]), test_idx))


def data_shuffle(x, y):
    # select 1
    #    x=np.array(x)
    #    y=np.array(y)
    #    r = np.random.permutation(len(y))
    #    x = x[r]
    #    y = y[r]
    # select 2
    np.random.seed(2019)
    np.random.shuffle(x)
    np.random.seed(2019)
    np.random.shuffle(y)
    return x, y


# X_train1, Y_train1 = X_train, Y_train
# X_train, Y_train = data_shuffle(X_train, Y_train)
# X_test1,Y_test1 = data_shuffle(X_test,Y_test)

in_shp = list(X_train.shape[1:])
print X_train.shape, in_shp
classes = mods
print classes

dr = 0.5 # dropout rate (%)
model = models.Sequential()
model.add(Reshape(in_shp + [1], input_shape=in_shp))#在网络第一步做了变换,in_shp---in_shp+[1],也可以对输入数据reshape
model.add(Conv2D(160, (5, 5), activation='relu'))
BatchNormalization(epsilon=1e-6, weights=None)
model.add(Conv2D(160, (5, 5), activation='relu'))
BatchNormalization(epsilon=1e-6, weights=None)
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu'))
BatchNormalization(epsilon=1e-6, weights=None)
model.add(Conv2D(64, (3, 3), activation='relu'))
BatchNormalization(epsilon=1e-6, weights=None)
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, kernel_initializer="glorot_normal", activation="relu", name="dense1"))
model.add(Dropout(dr))
# model.add(Dense(1, activation='sigmoid'))

model.add(Dense(len(classes), kernel_initializer="he_normal", name="dense2"))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
# plot_model(model, to_file='model.png',show_shapes=True)
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd)

batch_size = 128

filepath = 'model.h5'
history = model.fit(X_train,
                    Y_train,
                    batch_size=128,
                    epochs=3,
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks=[
                        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,mode='auto'),
                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                    ])
score = model.evaluate(X_test, Y_test, batch_size=128)
model.load_weights(filepath)

json_string = model.to_json()
open('model.json', 'w').write(json_string)

print score

plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.figure()

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

np.save('overall.npy',confnorm)

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
plt.show()

