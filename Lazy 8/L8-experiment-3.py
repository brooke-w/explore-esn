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

j = 24
k = 36

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

def runESNSeeded(p,a,dw,dfb,sfb,B,model):
    return model

def objective(trial, args, data, dataval):
    # Parameters (to tune)
    #N = trial.suggest_int('N', 10,100) For Jaeger et al's work we know 20 neurons was sufficient and we can always scale up
    np.seterr(all='warn')
    p = trial.suggest_uniform("p", 0.02, 1.0)
    a = trial.suggest_uniform("a", 0.02, 1.0)
    dw = trial.suggest_loguniform("dw", 0.10, 1.0)
    dfb = trial.suggest_uniform("dfb", 0.10, 1.0)
    din = trial.suggest_uniform("din", 0.0, 1.0)
    sin = trial.suggest_uniform("sin",0.0,2.0)
    sfb = trial.suggest_uniform("sfb",0.0,2.0)
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
    washout = 1000
    bestNRMSE = 1000000
    bestMAE = 100
    bestR2 = 0
    seedUsed = 100
    nrmse0, mae0, r20 = 0,0,0
    for step in range (0,10):
        np.seterr(all='warn')
        model.sv = 0
        model.generateW(seed)
        model.generateWin(seed)
        model.generateWfb(seed)
        
        model.train(input_u = None, teacher=data, washout=washout)
        model.sv = 1
        predicted = model.run(input_u=None, time=20000,washout=1000)
        
        if np.isnan(np.min(predicted)):
            print('Exceptionally bad generation of ESN. Aborting sub-trial. (2)')
            nrmse0 = 100
        else:
            nrmse0, mae0, r20 = getScores(dataval[1000:], predicted)
        if nrmse0 < bestNRMSE:
            bestNRMSE = nrmse0
            bestR2 = r20
            bestMAE = mae0
            seedUsed = seed
            
        seed = seed + 1
    trial.set_user_attr('seed', seedUsed)
    trial.set_user_attr('NRMSE', bestNRMSE)
    trial.set_user_attr('MAE', bestMAE)
    trial.set_user_attr('R2', bestR2)
    trial.set_user_attr('isU2Y', args.isU2Y)
    trial.set_user_attr('isY2Y', args.isY2Y)
    trial.set_user_attr('resFunc', args.resFunc)
    trial.set_user_attr('outFunc', args.outFunc)
    trial.set_user_attr('distribution', args.distribution)
    
    np.seterr(all='warn')
    return bestNRMSE

def main():
    #Generate Figure 8 Data
    t_all = np.linspace(0, 2*(23000/200), 23000)
    t = t_all[0:3000]
    t_val = t_all[3000:]
    omega = 1
    
    x = np.sin(2*math.pi*omega*t)
    x = x.reshape(-1,1)
    x_val = np.sin(2*math.pi*omega*t_val)
    x_val = x_val.reshape(-1,1)
    
    y = np.cos(math.pi*omega*t)
    y = y.reshape(-1,1)
    y_val = np.cos(math.pi*omega*t_val)
    y_val = y_val.reshape(-1,1)
    
    # plt.plot(x[0:200], y[0:200])
    # plt.ylim(-1.25, 1.25)
    # plt.xlim(-2,2)
    # plt.show()

    data = np.column_stack((x,y))
    data_val = np.column_stack((x_val,y_val))
    
    global j,k
    for i in range(j,k):
        df = pd.read_excel('Architecture.xlsx').iloc[i,:]
        np.random.seed(0)
        #Parameters that are unchanging during optimization
        args = Namespace(
            K = 0,  
            L = 2,
            N = 20,
            v = np.random.uniform(-0.01,0.01,(20000, 20)),
            sv = 0,
            outAlg = 1,  
            isBias = True,
            isU2Y = df.iloc[1],       #opt 1/2 architcture
            isY2Y = df.iloc[2],      #opt 3/2 architcture
            resFunc = df.iloc[3],        #opt 4/5 architcture
            outFunc = df.iloc[4],        #opt 6/7/8 architcture
            distribution = df.iloc[5],   #opt 9/A/B architcture
            isClassification = False
        )
        
        # Create a study name:
        study_name = 'L8-experiment-' + str(df[0])
        filename = study_name + ".pkl"
        
        if path.exists(filename):
            study = joblib.load(filename)
        else:
            study = optuna.create_study(study_name=study_name, sampler = optuna.samplers.TPESampler(seed=0))
        start = time.time() 
        # Optimize:
        study.optimize(lambda trial: objective(trial, args, data, data_val), n_trials=150)
        end = time.time()
        print("\n")
        print(end-start, "seconds")
        
        joblib.dump(study, filename)

if __name__ == "__main__":
    main()