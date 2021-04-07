#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:38:35 2021

@author: bweborg
"""

import numpy as np
import pandas as pd
import math
import time
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse

from ESN import ESN as esn

from argparse import Namespace
import joblib
import optuna
from os import path

j = 12
k = 24
def getScores(actual, predicted): 
    np.seterr(all='raise')
    try:
        mse0 = mse(actual, predicted)
        rmse = math.sqrt(mse0)
        nmrse = rmse / np.var(actual)
        
        mae0 = mae(actual, predicted)
        
        r20 = r2(actual, predicted)
        
    except FloatingPointError:
        print('Exceptionally bad generation of ESN. Aborting sub-trial. (1)')
        nmrse = 100
        mae0 = 100
        r20 = 0

    np.seterr(all='warn')
    return nmrse, mae0, r20

def narma10(x):
    """ tenth-order NARMA system applied to the input signal
    """
    size = len(x)
    y = np.zeros(x.shape)
    for n in range(10,size):
            y[n] = 0.3*y[n-1] + 0.05*y[n-1]*(y[n-1]+y[n-2]+y[n-3] +y[n-4]+y[n-5]+y[n-6]+y[n-7]+y[n-8]+y[n-9]+y[n-10])+ 1.5*x[n-10]*x[n-1] + 0.1
    return y

def get_esn_data(x,y,trainsize,testsize,inscale=1.,inshift=0.):
        """ returns trainin, trainout, testin, testout
        """
        skip = 50 # NARMA initialization
        trainin = x[skip:skip+trainsize]
        trainin.shape = 1,-1
        trainout = y[skip:skip+trainsize]
        trainout.shape = 1,-1
        testin = x[skip+trainsize:skip+trainsize+testsize]
        testin.shape = 1,-1
        testout = y[skip+trainsize:skip+trainsize+testsize]
        testout.shape = 1,-1
        return trainin, trainout, testin, testout

def objective(trial, args, trainin, trainout, testin, testout):
    # Parameters (to tune)
    p = trial.suggest_loguniform("p", 0.009, 1.0)
    a = trial.suggest_loguniform("a", 0.009, 1.0)
    dw = trial.suggest_loguniform("dw", 0.01, 1.0)
    dfb = trial.suggest_uniform("dfb", 0.0, 1.0)
    din = trial.suggest_uniform("din", 0.0, 1.0)
    sin = trial.suggest_uniform("sfb",0.0,2.0)
    B = trial.suggest_loguniform("B", 0.001, 2.0)

    model = esn(K = args.K,
                L = args.L,
                N = args.N,
                p = p,
                a = a,
                v = args.v,
                dw = dw,
                din = din,
                dfb = dfb,
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
    washout = 200
    bestNRMSE = 1000000
    bestMAE = 100
    bestR2 = 0
    seedUsed = 100
    for step in range (0,10):
        model.sv = 0
        model.generateW(seed)
        model.generateWin(seed)
        model.generateWfb(seed)
        seed = seed + 1
        
        model.train(input_u = trainin, teacher=trainout, washout=washout)
        model.sv = 1
        predicted = model.run(input_u=testin, time=testin.shape[0], washout=washout)
        
        if np.isnan(np.min(predicted)):
            print('Exceptionally bad generation of ESN. Aborting sub-trial. (2)')
            nmrse = 100
            return nmrse
        nrmse0, mae0, r20 = getScores(testout[washout:,:], predicted)
        if nrmse0 < bestNRMSE:
            bestNRMSE = nrmse0
            bestR2 = r20
            bestMAE = mae0
            seedUsed = seed
    
    trial.set_user_attr('seed', seedUsed)
    trial.set_user_attr('NRMSE', bestNRMSE)
    trial.set_user_attr('MAE', bestMAE)
    trial.set_user_attr('R2', bestR2)
    return bestNRMSE

def main():
    #Generate Data
    trainsize = 1200
    testsize = 2200
    
    # generate train/test signals
    size = trainsize+testsize+50
    x = (np.random.uniform(0,0.5, size)).reshape(-1,1) #from the text
    y = (narma10(x)).reshape(-1,1)               #from the text
    
    # create in/outs with bias input
    trainin, trainout, testin, testout = get_esn_data(x,y,trainsize,testsize)
    trainin = np.transpose(trainin)
    trainout = trainout.reshape(-1,1)
    testin = np.transpose(testin)
    testout = testout.reshape(-1,1)
    
    global j,k
    for i in range(j,k):
        df = pd.read_excel('Architecture.xlsx').iloc[i,:]
        
        #Parameters that are unchanging during optimization
        args = Namespace(
            K = 1,  
            L = 1,
            N = 100,                    #From text
            v = (np.random.uniform(-1,1, size)).reshape(-1,1), #from text
            sv = 0.0001,                #From text
            sfb = 0,                    #No feedback required
            outAlg = 1,  
            isBias = True,              
            isU2Y = df.iloc[1],         #opt 1/2 architcture
            isY2Y = df.iloc[2],         #opt 3/2 architcture
            resFunc = df.iloc[3],       #opt 4/5 architcture
            outFunc = df.iloc[4],       #opt 6/7/8 architcture
            distribution = df.iloc[5],  #opt 9/A/B architcture
            isClassification = False
        )
        
        # Create a study name:
        study_name = 'NARMA-experiment-' + str(df[0])
        filename = study_name + ".pkl"
        
        if path.exists(filename):
            study = joblib.load(filename)
        else:
            study = optuna.create_study(study_name=study_name, sampler = optuna.samplers.TPESampler(seed=0))
        start = time.time() 
        # Optimize:
        study.optimize(lambda trial: objective(trial, args, trainin, trainout, testin, testout), n_trials=150)
        end = time.time()
        print("\n")
        print(end-start, "seconds")
        
        joblib.dump(study, filename)

if __name__ == "__main__":
    main()