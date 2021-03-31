#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:38:35 2021

@author: bweborg
"""

import numpy as np
import matplotlib.pyplot as plt
import math

from ESN import ESN as esn

from argparse import Namespace
import joblib
import optuna
from os import path

def nrmse(actual, predicted):
    numSamples = actual.shape[0]
    mse = np.sum((predicted - actual)**2) / (numSamples)
    rmse = math.sqrt(mse)
    nmrse = rmse / np.var(actual)
    return nmrse

def objective(trial, args, data, dataval):
    # Parameters (to tune)
    N = trial.suggest_int('N', 10,100)
    p = trial.suggest_uniform("p", 0.0, 1.0)
    a = trial.suggest_uniform("a", 0.0, 1.0)
    dw = trial.suggest_loguniform("dw", 0.01, 1.0)
    dfb = trial.suggest_uniform("dfb", 0.0, 1.0)
    sfb = trial.suggest_uniform("sfb",0.10,2.0)
    B = trial.suggest_uniform("B", 0, 2.0)

    model = esn(K = args.K,
                L = args.L,
                N = N,
                p = p,
                a = a,
                v = args.v,
                dw = dw,
                din = args.din,
                dfb = dfb,
                sin = args.sin,
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
    bestScore = 100
    seedUsed = 100
    for step in range (0,20):
        model.generateW(seed)
        model.generateWin(seed)
        model.generateWfb(seed)
        seed = seed + 1
        
        model.train(input_u = None, teacher=data, washout=washout)
        predicted = model.run(input_u=None, time=20000,washout=1000)
        score = nrmse(dataval[1000:], predicted)
        if score < bestScore:
            bestScore = score
            seedUsed = seed
    
    trial.set_user_attr('seed', seedUsed)
    trial.set_user_attr('NRMSE', bestScore)
    return bestScore

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
    
    plt.plot(x[0:200], y[0:200])
    plt.ylim(-1.25, 1.25)
    plt.xlim(-2,2)
    plt.show()

    data = np.column_stack((x,y))
    data_val = np.column_stack((x_val,y_val))
    
    #Parameters that are unchanging during optimization
    args = Namespace(
        K = 0,
        L = 2,
        v = np.zeros(data.shape[0]+data_val.shape[0]),
        din = 0,
        sin = 0,
        sv = 0,
        resFunc = 1,
        outFunc = 0,
        outAlg = 1,
        distribution = 0,
        isBias = True,
        isU2Y = False,
        isY2Y = False,
        isClassification = False
    )
    
    # Create a study name:
    study_name = 'lazy-8'
    
    
    if path.exists('experiments.pkl'):
        study = joblib.load('experiments.pkl')
    else:
        study = optuna.create_study(study_name=study_name)
        
    # Optimize:
    study.optimize(lambda trial: objective(trial, args, data, data_val), n_trials=10)
    
    joblib.dump(study, 'experiments.pkl')

if __name__ == "__main__":
    main()