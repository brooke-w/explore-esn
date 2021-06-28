#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:38:35 2021

@author: bweborg
"""

import numpy as np
import pandas as pd
import time
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as rocAuc
from sklearn.metrics import accuracy_score

from ESN import ESN as esn

from argparse import Namespace
import joblib
import optuna
from os import path

j = 0
k = 12

def getScores(actual, predicted): 
    np.seterr(all='raise')
    try:
        f1Score = f1(actual, predicted, average='samples')
        
        aucScore = rocAuc(actual, predicted)
        
        accuracyScore = accuracy_score(actual, predicted)
        
    except FloatingPointError:
        print('Exceptionally bad generation of ESN. Aborting sub-trial. (1)')
        f1Score = -1
        aucScore = -1
        accuracyScore = -1
        
    except ValueError:
        print('Exceptionally bad generation of ESN. Aborting sub-trial. (3)')
        f1Score = -1
        aucScore = -1
        accuracyScore = -1

    np.seterr(all='warn')
    return f1Score, aucScore, accuracyScore

def ttvSplit(data, numClasses):
    np.random.seed(0)
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

def objective(trial, args, trainin, trainout, testin, testout):
    # Parameters (to tune)
    #N = trial.suggest_int('N', 10,100) For Jaeger et al's work we know 20 neurons was sufficient and we can always scale up
    np.seterr(all='warn')
    p = trial.suggest_uniform("p", 0.02, 1.0)
    a = trial.suggest_uniform("a", 0.02, 1.0)
    dw = trial.suggest_loguniform("dw", 0.10, 0.5)
    din = trial.suggest_uniform("din", 0.10, 1.0)
    sin = trial.suggest_loguniform("sin",0.05,2.0)
    B = trial.suggest_loguniform("B", 1e-9, 2.0)

    model = esn(K = args.K,
                L = args.L,
                N = args.N,
                p = p,
                a = a,
                v = args.v,
                dw = dw,
                din = din,
                dfb = args.dfb,
                sin = sin,
                sfb = args.sfb,
                sv = args.sv,
                resFunc = args.resFunc,
                outFunc = args.outFunc,
                outAlg = args.outAlg,
                B = B,
                distribution = args.distribution,
                isBias = args.isBias,
                isU2Y = args.isU2Y,
                isY2Y = args.isY2Y,
                isClassification = args.isClassification)
    
    seed = 100
    washout = 10
    bestF1 = -1
    bestAuc = -1
    bestAccuracy = -1
    seedUsed = 100
    f1Score, aucScore, accuracyScore = 0,0,0
    for step in range (0,10):
        np.seterr(all='warn')
        model.generateW(seed)
        model.generateWin(seed)
        model.generateWfb(seed)
        
        model.train(input_u = trainin, teacher=trainout, washout=washout) #zero start state is default
                
        predicted = model.run(input_u=testin, time=testin.shape[0], washout=washout)
        
        if np.isnan(np.min(predicted)):
            print('Exceptionally bad generation of ESN. Aborting sub-trial. (2)')
            f1Score = -1
        else:
            f1Score, aucScore, accuracyScore = getScores(testout, predicted)
        if f1Score > bestF1:
            bestF1 = f1Score
            bestAuc = aucScore
            bestAccuracy = accuracyScore
            seedUsed = seed
            
        seed = seed + 1
    trial.set_user_attr('seed', seedUsed)
    trial.set_user_attr('F1', bestF1)
    trial.set_user_attr('AUC', bestAuc)
    trial.set_user_attr('Accuracy', bestAccuracy)
    trial.set_user_attr('isU2Y', args.isU2Y)
    trial.set_user_attr('isY2Y', args.isY2Y)
    trial.set_user_attr('resFunc', args.resFunc)
    trial.set_user_attr('outFunc', args.outFunc)
    trial.set_user_attr('distribution', args.distribution)
    
    np.seterr(all='warn')
    return bestF1

def main():
    #Generate Data
    numClasses = 5
    df = pd.read_csv('data.csv')
    df = df.loc[df.iloc[:,-1].isin(np.arange(0,numClasses))]
    df = df.to_numpy()
    df = esnClassRep(df,numClasses)
    trainin, trainout, testin, testout, valin, valout = ttvSplit(df, numClasses)
    testout = compressTargets(testin, testout)
    trainoutT = np.tanh(trainout) #The inverse arctanh function is not defined for 1 so transforming the outputs allows use of tanh reservoir activation for training
    #We don't have to undo the transform because the ESN classifier automatically rounds and returns one for the best choice class
    
    global j,k
    for i in range(j,k):
        df = pd.read_excel('Architecture.xlsx').iloc[i,:]
        np.random.seed(0)
        #Parameters that are unchanging during optimization
        args = Namespace(
            K = 85,  
            L = numClasses,
            N = 50,                     #Text used 200
            v = (np.random.uniform(-1,1, trainin.shape[0])),
            sv = 0,                     #From text
            sfb = 0,                    #No feedback required
            dfb = 0,
            outAlg = 1,  
            isBias = True,              
            isU2Y = df.iloc[1],         #opt 1/2 architcture
            isY2Y = df.iloc[2],         #opt 3/2 architcture
            resFunc = df.iloc[3],       #opt 4/5 architcture
            outFunc = df.iloc[4],       #opt 6/7/8 architcture
            distribution = df.iloc[5],  #opt 9/A/B architcture
            isClassification = True
        )
        
        # Create a study name:
        study_name = 'Digit-experiment-' + str(df[0])
        filename = study_name + ".pkl"
        
        if path.exists(filename):
            study = joblib.load(filename)
        else:
            study = optuna.create_study(study_name=study_name, sampler = optuna.samplers.TPESampler(seed=0), direction='maximize')
        start = time.time() 
        # Optimize:
        study.optimize(lambda trial: objective(trial, args, trainin, trainoutT, testin, testout), n_trials=50)
        end = time.time()
        print("\n")
        print(end-start, "seconds")
        
        joblib.dump(study, filename)

if __name__ == "__main__":
    main()