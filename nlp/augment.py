# -*- coding: utf-8 -*-
#数据增强，shuffle或drop
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

train_df = pd.read_csv('../input/train.csv')
#train_df["question_text"] = train_df["question_text"].map(lambda x: clean_text(x))

test_df = pd.read_csv('../input/test.csv')
#test_df["question_text"] = test_df["question_text"].map(lambda x: clean_text(x))

train_df = train_df[1:40]
test_df = test_df[1:40]
X_train = train_df["question_text"].fillna("na").values
X_test = test_df["question_text"].fillna("na").values
y = train_df["target"]

xtrain = list(X_train)
xtest = list(X_test)
yl = list(y.values)
def shuffle(d):
    return np.random.permutation(d)

def shuffle2(d):
    len_ = len(d)
    times = 2
    for i in range(times):
        index = np.random.choice(len_, 2)
        d[index[0]],d[index[1]] = d[index[1]],d[index[0]]
    return d

def dropout(d, p=0.4):
    len_ = len(d)
    index = np.random.choice(len_, int(len_ * p))
    for i in index:
        d[i] = ' '
    return d


def clean(xx):
    xx2 = re.sub(r'\?', "", xx)
    xx1= xx2.split(' ')
    return xx1


def dataaugment(X,y):
    l = len(X)
    for i in range(l):
        item = clean(X[i])
        d1 = shuffle2(item)
        d11 = ' '.join(d1)
        d2 = dropout(item)
        d22 = ' '.join(d2)
        X.extend([d11,d22])
        y.extend([y[i],y[i]])
    return X,y

Xp, yp = dataaugment(xtrain, yl)