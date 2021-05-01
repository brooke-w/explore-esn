#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:20:47 2021

@author: bweborg
"""
import pandas as pd
import numpy as np
from sklearn.metrics.classification import f1_score as f1
from ESN import ESN as esn

#Shuffle the data

#Takes a data set and splits it into test, training, and validation sets
#60-20-20 split
#Samples w/ same specimen will be kept together in one of the three sets

#If all the classes are in a single column, this function
#will break up that representation into multiple columns
#for the ESN
def esnClassRep(data, uniqueClasses):
    targets = np.zeros((data.shape[0],uniqueClasses))
    
    for i in range(0, data.shape[0]):
        rowTarget = int(data[i,-1])
        targets[i,rowTarget] = 1
        
    data = data[:,:-1]
    data = np.concatenate((data,targets), axis=1)
    return data

def compressTargets(features, output):
    numTimeSeq = np.unique(features[:,0]).shape[0]    #get number of unique time sequences
    _, groupID = np.unique(features[:,0], return_index=True)
    groupID = features[np.sort(groupID), 0]
    
    indexer = 0
    r = np.zeros((numTimeSeq, output.shape[1]))
    for i in groupID:
        mask = (features[:, 0] == i)         #grab all rows that have groupID i
        r[indexer,:] = (output[mask,:])[0]     #we don't need to average this 
        indexer = indexer + 1
    return r
    

def ttvSplit(data, numClasses):
    data = pd.DataFrame(data)
    numFeatures = data.shape[1] - numClasses
    features = pd.DataFrame(data.iloc[:,:numFeatures])
    targets = data.iloc[:,numFeatures:numFeatures+numClasses]

    groups = pd.Series(features.iloc[:,0].unique()).sample(frac=1).reset_index(drop=True)
    ind0 = round(groups.shape[0] * .6)
    ind1 = round(groups.shape[0] * .2) + ind0

    trainingGroups = groups.iloc[0:ind0]
    testGroups = groups.iloc[ind0:ind1]
    validGroups = groups.iloc[ind1:]

    training = features.loc[features.iloc[:,0].isin(trainingGroups.values)]
    test = features.loc[features.iloc[:,0].isin(testGroups.values)]
    validation = features.loc[features.iloc[:,0].isin(validGroups.values)]

    trainingT = targets.loc[targets.index.isin(training.index)]
    testT = targets.loc[targets.index.isin(test.index)]
    validT = targets.loc[targets.index.isin(validation.index)]

    return training.to_numpy(), trainingT.to_numpy(), test.to_numpy(), testT.to_numpy(), validation.to_numpy(), validT.to_numpy()

np.random.seed(0)
df = pd.read_csv('data.csv')
#df = df.loc[df.iloc[:,-1].isin([0,9])]
df = df.to_numpy()
df = esnClassRep(df,10)
trainin, trainout, testin, testout, valin, valout = ttvSplit(df, 10)

model = esn(K = 85,
            L = 10,
            N = 200,
            p = 0.7,
            a = 1,
            v = (np.random.uniform(-1,1, df.shape[0])).reshape(-1,1),
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

seed = 3#np.random.randint(0,100)
washout = 10
model.generateW(seed)
model.generateWin(seed)
model.generateWfb(seed)

model.train(input_u = trainin, teacher=trainout, washout=washout) #zero start state is default
        
predicted = model.run(input_u=testin, time=testin.shape[0], washout=washout)

testout = compressTargets(testin, testout)
score = f1(testout, predicted, average='samples')
print(score)
print(seed)