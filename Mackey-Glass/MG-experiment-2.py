#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:53:13 2021

@author: bweborg
"""
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
import matplotlib.pyplot as plt

j = 12
k = 24

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

def objective(trial, args, trainin, testin):
    # Parameters (to tune)
    #N = trial.suggest_int('N', 10,100) For Jaeger et al's work we know 20 neurons was sufficient and we can always scale up
    np.seterr(all='warn')
    p = trial.suggest_uniform("p", 0.02, 1.0)
    a = trial.suggest_loguniform("a", 0.02, 1.0)
    dw = trial.suggest_loguniform("dw", 0.01, 1.0)
    din = trial.suggest_uniform("din", 0.0, 1.0)
    sin = trial.suggest_uniform("sin",0.0,2.0)
    dfb = trial.suggest_uniform("dfb", 0.0, 1.0)
    sfb = trial.suggest_uniform("sfb",0.0,2.0)
    B = trial.suggest_loguniform("B", 1e-9, 2.0)

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
                sfb = sfb,
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
    washout = 100
    bestRMSE = 1000000
    bestMAE = 100
    bestR2 = 0
    seedUsed = 100
    rmse0, mae0, r20 = 0,0,0
    for step in range (0,10):
        np.seterr(all='warn')
        '''
        The network was run from a zero starting state in 
        teacher-forced mode with the τ = 17, length 3000 training sequence. 
        During this run, noise was inserted... The first 1000 steps were 
        discarded and the output weights were computed by a linear regression from the remaining 2000 network states.'''
        model.generateW(seed)
        model.generateWin(seed)
        model.generateWfb(seed)
        
        model.train(input_u = None, teacher=trainin, washout=washout) #zero start state is default
        
        '''the trained network was run for 4000 steps. The first 1000 steps were teacher-forced with 
        a newly generated sequence from the original system. The network output of the remaining 
        free-running 3000 steps was re-transformed to the original coordi- nates by y 􏰀→ arctanh(y) + 1.'''
        
        #doesn't say to remove noise so I left it
        predicted = model.run(input_u=None, time=testin.shape[0], washout=washout)
        transformed = np.arctan(predicted) + 1
        
        if np.isnan(np.min(transformed)):
            print('Exceptionally bad generation of ESN. Aborting sub-trial. (2)')
            rmse0 = 100
        else:
            rmse0, mae0, r20 = getScores(trainin[washout:], transformed)
        if rmse0 < bestRMSE:
            bestRMSE = rmse0
            bestR2 = r20
            bestMAE = mae0
            seedUsed = seed
            
        seed = seed + 1
    trial.set_user_attr('seed', seedUsed)
    trial.set_user_attr('rmse', bestRMSE)
    trial.set_user_attr('MAE', bestMAE)
    trial.set_user_attr('R2', bestR2)
    trial.set_user_attr('isU2Y', args.isU2Y)
    trial.set_user_attr('isY2Y', args.isY2Y)
    trial.set_user_attr('resFunc', args.resFunc)
    trial.set_user_attr('outFunc', args.outFunc)
    trial.set_user_attr('distribution', args.distribution)
    
    np.seterr(all='warn')
    return bestRMSE

def main():
    #Generate Data
    trainsize = 2000
    testsize = 2000
    
    # generate train/test signals
    y = (np.array(mg17(trainsize+testsize+100))).reshape(-1,1)
    #Transform
    transformedY = np.tanh(y-1)
    trainin = transformedY[:trainsize,:]
    testin = y[trainsize:,:]
    
    # z = np.arange(0, 500)

    # fig, ax = plt.subplots()
    # ax.plot(z, y[0:500,0])
    # ax.set(xlabel='Time Step', 
    #        ylabel='Mackey-Glass 17',
    #        title='Mackey-Glass Training Data')
    # ax.legend()

    
    global j,k
    for i in range(j,k):
        df = pd.read_excel('Architecture.xlsx').iloc[i,:]
        np.random.seed(0)
        #Parameters that are unchanging during optimization
        args = Namespace(
            K = 0,  
            L = 1,
            N = 300,                    
            v = (np.random.uniform(-1,1, testsize)).reshape(-1,1), #from text
            sv = 0,                     #Don't need using ridge
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
        study_name = 'MG-experiment-' + str(df[0])
        filename = study_name + ".pkl"
        
        if path.exists(filename):
            study = joblib.load(filename)
        else:
            study = optuna.create_study(study_name=study_name, sampler = optuna.samplers.TPESampler(seed=0))
        start = time.time() 
        # Optimize:
        study.optimize(lambda trial: objective(trial, args, trainin, testin), n_trials=150)
        end = time.time()
        print("\n")
        print(end-start, "seconds")
        
        joblib.dump(study, filename)

if __name__ == "__main__":
    main()