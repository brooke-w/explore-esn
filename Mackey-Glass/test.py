#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:51:53 2021

@author: bweborg
"""
import numpy as np
from ESN import ESN as esn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
import math

def mg17(length):
    # https://towardsdatascience.com/learn-ai-today-04-time-series-multi-step-forecasting-6eb48bbcc724
    alpha = 0.2
    beta = 10
    gamma = 0.1
    tau = 17
    
    y = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485,
     1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759]

    for n in range(17,length+99):
        y.append(y[n] - gamma*y[n] + alpha*y[n-tau]/(1+y[n-tau]**beta))
    y = y[100:]
    
    return y[100:]

def getScores(actual, predicted): 
    np.seterr(all='raise')
    try:
        mse0 = mse(actual, predicted)
        rmse = math.sqrt(mse0)
        
        mae0 = mae(actual, predicted)
        
        r20 = r2(actual, predicted)
        
    except FloatingPointError:
        print('Exceptionally bad generation of ESN. Aborting sub-trial. (1)')
        rmse = 100
        mae0 = 100
        r20 = 0
        
    except ValueError:
        print('Exceptionally bad generation of ESN. Aborting sub-trial. (3)')
        rmse = 100
        mae0 = 100
        r20 = 0

    np.seterr(all='warn')
    return rmse, mae0, r20

#Generate Data
trainsize = 2000
testsize = 2000

# generate train/test signals
y = (np.array(mg17(trainsize+testsize+100))).reshape(-1,1)
#Transform
transformedY = np.tanh(y-1)
trainin = transformedY[:trainsize,:]
testin = y[trainsize:,:]

'''signed values of 0, 0.4, −0.4 with probabilities 0.9875, 0.00625, 0.00625. 
This resulted in a weight matrix W with a sparse connectivity of 1.25%.

One input unit was attached which served to feed a constant bias signal u(n) = 0.2 
into the network. The input connections were randomly chosen to be 0, 0.14, −0.14 
with probabilities 0.5, 0.25, 0.25.

One output unit was attached with output feedback connections sampled randomly 
from the uniform distribution over [−0.56, 0.56].'''
np.random.seed(42)
model = esn(K = 0,
            L = 1,
            N = 300,
            p = 1.25,
            a = 0.3, #or 0.9?
            v = (np.zeros(testsize)).reshape(-1,1),
            dw = 0.0125,
            din = 0.50,
            dfb = 1.00,
            sin = 0.5,
            sfb = 1,
            sv = 0.00001,
            resFunc = 1,
            outFunc = 0,
            outAlg = 1,                 #MPI
            B = 0.000001,
            distribution = 1,           
            isBias = True,
            isU2Y = False,
            isY2Y = False,
            isClassification = False)

'''
The network was run from a zero starting state in 
teacher-forced mode with the τ = 17, length 3000 training sequence. 
During this run, noise was inserted... The first 1000 steps were 
discarded and the output weights were computed by a linear regression from the remaining 2000 network states.'''
seed=42
washout = 100
model.generateW(seed)
model.generateWin(seed)
model.generateWfb(seed)

model.train(input_u = None, teacher=trainin, washout=washout) #zero start state is default

'''the trained network was run for 4000 steps. The first 1000 steps were teacher-forced with 
a newly generated sequence from the original system. The network output of the remaining 
free-running 3000 steps was re-transformed to the original coordi- nates by y 􏰀→ arctanh(y) + 1.'''

#doesn't say to remove noise so I left it
predicted = model.run(input_u=None, time=testin.shape[0])
transformed = np.arctan(predicted) + 1

rmse0, mae0, r20 = getScores(testin[:,:], transformed)
print(rmse0)
print(r20)
print(mae0)

z = np.arange(0, testsize)
fig, ax = plt.subplots()
ax.plot(z, transformed[:,0], 'b', label='Predicted(w/ Transform)')
ax.plot(z, testin[:,0], 'r',label='actual')
ax.set(xlabel='Time Step', 
       ylabel='Mackey-Glass 17',
       title='Mackey-Glass Training Data')
ax.legend()