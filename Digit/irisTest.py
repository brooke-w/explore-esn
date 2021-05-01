#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:23:48 2021

@author: bweborg
"""
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from ESN import ESN as esn
from sklearn.preprocessing import normalize as norm
from sklearn.metrics.classification import f1_score as f1

iris = datasets.load_iris()

X = np.array(iris.data[:, :])  # we only take the first two features.
y = np.array(iris.target)

#normalize input
x = norm(X)
#add groupID
groupID = np.arange(0,x.shape[0]).reshape(-1,1)
x = np.concatenate((groupID, x), axis = 1)

#Transform iris target data for ESN use
target = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in range(0,y.shape[0]):
    target[i,y[i]] = 1
    
shuffle = np.arange(0,X.shape[0])
np.random.shuffle(shuffle)
x = x[shuffle,:]
target = target[shuffle,:]

trainin = x[0:90,:]
trainout = target[0:90,:]
testin = x[90:,:]
testout = target[90:,:]
    
model = esn(K = 4,
            L = 3,
            N = 20,
            p = 0.2,
            a = 1,
            v = (np.random.uniform(-1,1, 150)).reshape(-1,1),
            dw = 0.15,
            din = 1,
            dfb = 0,
            sin = 1,
            sfb = 0,
            sv = 0,
            resFunc = 1,
            outFunc = 0,
            outAlg = 1,
            B = 0.0001,
            distribution = 0,
            isBias = False,
            isU2Y = True,
            isY2Y = False,
            isClassification = True)

seed = 100
washout = 0
model.generateW(seed)
model.generateWin(seed)
model.generateWfb(seed)

model.train(input_u = trainin, teacher=trainout, washout=washout) #zero start state is default
        
predicted = model.run(input_u=testin, time=testin.shape[0])
score = f1(testout, predicted, average='samples')
print(score)