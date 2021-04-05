#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: brookeweborg

Reservoir Computing Class
Controls the training and prediction of a reservoir computing model
"""

import numpy as np
import networkx as nx
import random as rand

class ESN:
    def __init__(self, 
                 K,                                 #Input units
                 L,                                 #Output units
                 N = 50,                            #Neurons in the reservoir
                 p = 0.9,                           #Spectral radius
                 a = 0.50,                          #Leaky rate - set to 1 for non-leaky neuron
                 v = 0.06,                          #Injected noise - usually can just feed it a uniform random array with values [-1,1] use sv to scale
                 dw = 0.5,                          #density of connections in W (the reservoir)
                 din = 0.95,                        #density of connections from inputs into reservoir
                 dfb = 0,                           #density of connections from outputs back into reservoir
                 sin = 1,                           #scaling value for Win
                 sfb = 1,                           #scaling value for Wfb
                 sv = 1,                            #scaling value for noise, v
                 resFunc = 1,                       #Value represents what function is to be used for reservoir activation
                 outFunc = 0,                       #Value represents what function is to be used for output activation
                 outAlg = 0,                        #Which algorithm to calculate Wout
                 B = 1,                             #Beta value for Ridge Regression if used for calculating Wout
                 distribution = 0,                  #Value represents which type of weight distribution to use
                 isBias = False,                    #If true, a bias value of 1 will automatically be added to the input vector,
                 isU2Y = True,                      #If true, a connection will be made directly from the input to the output for the prediction calculation
                 isY2Y = False,                     #If true, a connection will be made directly from the output to the output for the prediction calculation
                 isClassification=False):           #Train an ESN classifier (default is regression)   
        #assignments to object
        self.W = None
        self.Win = None
        self.Wfb = None
        self.K = K
        self.L = L
        self.N = N
        self.p = p
        self.a = a
        self.v = v
        self.dw = dw
        self.din = din
        self.dfb = dfb
        self.sin = sin
        self.sfb = sfb
        self.isBias = isBias
        self.outAlg = outAlg
        self.B = B
        self.distribution = distribution
        self.isU2Y = isU2Y
        self.isY2Y = isY2Y
        self.isClassification = isClassification
        
        #setting activation functions
        self.resFunc = self.setFunc(resFunc)[0]
        self.outFunc, self.invFunc = self.setFunc(outFunc)
        
        if isBias:
            self.K = self.K + 1
            
        self.T = None               #holds teacher data
        self.M = None              #holds reservour states
        self.Wout = None           #holds output weights
        
        #Generate all the weights for the model
        self.generateW()
        self.generateWin()
        self.generateWfb()
        
        return
    
 #  ______  _____ _   _    _____      _               
 # |  ____|/ ____| \ | |  / ____|    | |              
 # | |__  | (___ |  \| | | (___   ___| |_ _   _ _ __  
 # |  __|  \___ \| . ` |  \___ \ / _ \ __| | | | '_ \ 
 # | |____ ____) | |\  |  ____) |  __/ |_| |_| | |_) |
 # |______|_____/|_| \_| |_____/ \___|\__|\__,_| .__/ 
 #                                             | |    
 #                                             |_|    
    
    '''This function will return a lambda function and its inverse based on a given number
    Precondition: give a number between 0 and 2
    Postcondition: returns a lambda function, and another lambda function that is the inverse of the first returned'''
    def setFunc(self, func):

        if func == 0:
            actFunc = lambda x: x                #linear function
            invFunc = actFunc                    #inverse is itself because we didn't actually change the input
            return actFunc, invFunc
        elif func == 1:
            actFunc = lambda x: np.tanh(x)       #tanh == 1
            invFunc = lambda x: np.arctanh(x)
            return actFunc, invFunc
        elif func == 2:
            actFunc = lambda x: np.sinc(x)       #sinc == 2
            #sinc has no true inverse but an approximation
            invFunc = lambda x: (1 - 0.035*(x**2)) / (1 + 0.15*(x**2) - 0.018*(x**4)) #https://math.stackexchange.com/questions/2175174/inverse-sinc-approximation
            return actFunc, invFunc
            
        return
            
    
    '''This function generates the reservoir aka the weight matrix W.
    Precondition: all the values for the ESN object have been set
    Postcondition: Weights are saved to the object as W'''
    def generateW(self, seed = None):
        if seed is not(None):
            rand.seed = seed
            np.random.seed(seed)
        
        #random graph
        #N is the number of nodes in the graph and dw is the density or probability a connection is created        
        maxEigen = 0
        while(maxEigen == 0):
            G = nx.gnp_random_graph(self.N, self.dw, directed = True)
            W = np.zeros((self.N, self.N))
            for edge in G.edges:
                row = edge[0]
                col = edge[1]
                W[row, col] = 1
            #we need to rescale the entries in the matrix to have negative and positive connections
            #use continuous method
            #multiply each non-zero entry by random number in uniform distribution -sigma_1 to sigma_1
            if self.distribution == 0:                              #uniform
                for i in range(0,self.N):
                    for j in range(0,self.N):
                            W[i, j] = W[i, j] * rand.uniform(-1,1)
            elif self.distribution == 1:                            #discerete bi-valued
                for i in range(0,self.N):
                    for j in range(0,self.N):
                        prob = rand.uniform(0,1)
                        if prob < 0.5:
                            W[i, j] = W[i, j] * -1
                        #else it's one
            elif self.distribution == 2:                            #Laplace
                d = np.random.laplace(0, 1, (self.N,self.N))
                W = np.multiply(W,d)                                #multiply arguments element-wise
            
            
            #get eigenvalues of W
            eigenVal, eigenValScaled = np.linalg.eig(W)
            
            #get absolute max of eigenvalues
            index = np.argmax(eigenVal)
            maxEigen0 = np.abs(eigenVal[index])
            index = np.argmin(eigenVal)
            maxEigen1 = np.abs(eigenVal[index])
            if maxEigen0 > maxEigen1:
                maxEigen = maxEigen0
            else:
                maxEigen = maxEigen1
            
        
        #Scaling of Weight matrix
        W = (self.p / maxEigen) * W
        self.W = W
        return
    
    '''This function generates the inputs weight connections Win.
    Precondition: all the values for the ESN object have been set
    Postcondition: Weights are saved to the object as Win'''
    def generateWin(self, seed = None):
        if seed is not(None):
            rand.seed = seed
            np.random.seed(seed)
        
        #create NxK array of zeros
        Win = np.zeros((self.N, self.K))
        for i in range(0,self.N):
            for j in range(0,self.K):
                prob = rand.uniform(0,1)
                if prob < self.din:
                    Win[i, j] = 1
        
        #we need to rescale the entries in the matrix to have negative and positive connections
        #use continuous method
        #multiply each non-zero entry by random number in uniform distribution -sigma_1 to sigma_1
        if self.distribution == 0:                              #uniform
            randomVals = np.random.uniform(-1,1, (self.N,self.K))
            Win = Win * randomVals            #multiply arguments element-wise
        elif self.distribution == 1:                            #discerete bi-valued
            for i in range(0,self.N):
                for j in range(0,self.K):
                    prob = rand.uniform(0,1)
                    if prob < 0.5:
                        Win[i, j] = -1
        elif self.distribution == 2:                            #Laplace
            d = np.random.laplace(0, 1, (self.N,self.K))
            Win = Win*d                              #multiply arguments element-wise
        
        self.Win = self.sin * Win
        return
    
    '''This function generates the inputs weight connections Win.
    Precondition: all the values for the ESN object have been set
    Postcondition: Weights are saved to the object as Wfb'''
    def generateWfb(self, seed = None):
        if seed is not(None):
            rand.seed = seed
            np.random.seed(seed)
        
        #create NxL array of zeros
        Wfb = np.zeros((self.N, self.L))
        for i in range(0, self.N):
            for j in range(0, self.L):
                prob = rand.uniform(0,1)
                if prob < self.dfb:
                    Wfb[i, j] = 1
                    
        #we need to rescale the entries in the matrix to have negative and positive connections
        #use continuous method
        #multiply each non-zero entry by random number in uniform distribution -sigma_1 to sigma_1
        if self.distribution == 0:                              #uniform
            Wfb = np.multiply(Wfb, np.random.uniform(-1,1, (self.N,self.L))) #multiply arguments element-wise
        elif self.distribution == 1:                            #discerete bi-valued
            for i in range(0,self.N):
                for j in range(0,self.L):
                    prob = rand.uniform(0,1)
                    if prob < 0.5:
                        Wfb[i, j] = Wfb[i, j] * -1
                    #else it's one
        elif self.distribution == 2:                            #Laplace
            d = np.random.laplace(0, 1, (self.N,self.L))
            Wfb = np.multiply(Wfb,d)                               #multiply arguments element-wise
                    
        self.Wfb = self.sfb * Wfb
        return


#  _______        _       _               __  __      _   _               _     
# |__   __|      (_)     (_)             |  \/  |    | | | |             | |    
#    | |_ __ __ _ _ _ __  _ _ __   __ _  | \  / | ___| |_| |__   ___   __| |___ 
#    | | '__/ _` | | '_ \| | '_ \ / _` | | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
#    | | | | (_| | | | | | | | | | (_| | | |  | |  __/ |_| | | | (_) | (_| \__ \
#    |_|_|  \__,_|_|_| |_|_|_| |_|\__, | |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
#                                  __/ |                                        
#                                 |___/                                         


    ''''input and previous output are not used in calculating prediction'''
    def trainBasic(self, time, input_u, teacher, r, x, y, washout):
        resFunc = self.resFunc                                           #reservoir activation function
        invFunc = self.invFunc                                           #inverse of the output activation function needed for training
        
        M = np.zeros((time-washout, self.N))                      

        for t in range(0,time):
            u = (input_u[t,:]).reshape(-1,1)
            WdotX = (self.W).dot(x)
            WinDotU = (self.Win).dot(u)
            WfbDotY = (self.Wfb).dot(y)
            innerTerm = WdotX + WinDotU + WfbDotY + (self.sv*self.v[t]).reshape(-1,1)
            theTanTerm = resFunc(innerTerm)
            secondTerm = self.a * theTanTerm
            x = (1 - self.a) * x + secondTerm
            if t >= washout:
                k = t - washout
                M[k,:] = np.transpose(x)
                r[k,:] = invFunc(teacher[t, :])
            y = (teacher[t,:]).reshape(-1,1)
        self.M = M
        self.T = r
        return
    
        '''input and previous output are not used in calculating prediction'''
    def trainClassification(self, time, input_u, teacher, r, x, y, washout):
        resFunc = self.resFunc                                           #reservoir activation function
        invFunc = self.invFunc                                           #inverse of the output activation function needed for training
        
        M = np.zeros((time-washout, self.N))                         

        for t in range(0,time):
            u = (input_u[t,:]).reshape(-1,1)
            WdotX = (self.W).dot(x)
            WinDotU = (self.Win).dot(u)
            WfbDotY = (self.Wfb).dot(y)
            innerTerm = WdotX + WinDotU + WfbDotY + (self.sv*self.v[t]).reshape(-1,1)
            theTanTerm = resFunc(innerTerm)
            secondTerm = self.a * theTanTerm
            x = (1 - self.a) * x + secondTerm
            if t >= washout:
                k = t - washout
                M[k,:] = np.transpose(x)
                r[k,:] = invFunc(teacher[t, :])
            y = (teacher[t,:]).reshape(-1,1)
        self.M = M
        self.T = r
        return
    
    '''input is connected to output units'''
    def trainU2Y(self, time, input_u, teacher, r, x, y, washout):   
        resFunc = self.resFunc                                           #reservoir activation function
        invFunc = self.invFunc                                           #inverse of the output activation function needed for training
        M = np.zeros((time-washout, self.N+self.K))                      
        
        for t in range(0,time):
            u = (input_u[t,:]).reshape(-1,1)
            WdotX = (self.W).dot(x)
            WinDotU = (self.Win).dot(u)
            WfbDotY = (self.Wfb).dot(y)
            innerTerm = WdotX + WinDotU + WfbDotY + (self.sv*self.v[t]).reshape(-1,1)
            theTanTerm = resFunc(innerTerm)
            secondTerm = self.a * theTanTerm
            x = (1 - self.a) * x + secondTerm
            if t >= washout:
                k = t - washout
                ux = np.concatenate((u, x), axis=0)
                M[k,:] = np.transpose(ux)
                r[k,:] = invFunc(teacher[t, :])
            y = (teacher[t,:]).reshape(-1,1)
        self.M = M
        self.T = r
        return
    
    '''input is connected to output units and there are self recurrent connections into the output unit'''
    def trainUY2Y(self, time, input_u, teacher, r, x, y, washout):   
        resFunc = self.resFunc                                           #reservoir activation function
        invFunc = self.invFunc                                           #inverse of the output activation function needed for training
        M = np.zeros((time-washout, self.N+self.K+self.L))                         
        
        for t in range(0,time):
            u = (input_u[t,:]).reshape(-1,1)
            WdotX = (self.W).dot(x)
            WinDotU = (self.Win).dot(u)
            WfbDotY = (self.Wfb).dot(y)
            innerTerm = WdotX + WinDotU + WfbDotY + (self.sv*self.v[t]).reshape(-1,1)
            theTanTerm = resFunc(innerTerm)
            secondTerm = self.a * theTanTerm
            x = (1 - self.a) * x + secondTerm
            if t >= washout:
                k = t - washout
                uxy = np.concatenate((u, x, y), axis=0)
                M[k,:] = np.transpose(uxy)
                r[k,:] = invFunc(teacher[t, :])
            y = (teacher[t,:]).reshape(-1,1)
        self.M = M
        self.T = r
        return
    
    '''self recurrent connections into the output unit'''
    def trainY2Y(self, time, input_u, teacher, r, x, y, washout):   
        resFunc = self.resFunc                                           #reservoir activation function
        invFunc = self.invFunc                                           #inverse of the output activation function needed for training
        M = np.zeros((time-washout, self.N+self.L))                         
        
        for t in range(0,time):
            u = (input_u[t,:]).reshape(-1,1)
            WdotX = (self.W).dot(x)
            WinDotU = (self.Win).dot(u)
            WfbDotY = (self.Wfb).dot(y)
            innerTerm = WdotX + WinDotU + WfbDotY + (self.sv*self.v[t]).reshape(-1,1)
            theTanTerm = resFunc(innerTerm)
            secondTerm = self.a * theTanTerm
            x = (1 - self.a) * x + secondTerm
            if t >= washout:
                k = t - washout
                xy = np.concatenate((x, y), axis=0)
                M[k,:] = np.transpose(xy)
                r[k,:] = invFunc(teacher[t, :])
            y = (teacher[t,:]).reshape(-1,1)
        self.M = M
        self.T = r
        return

    
    '''Call this function with the ESN object to train the model;
    Precondition: ESN object has been instatiated and all appropriate parameters have been set.
    User must supply training input_u that has K features. Note: if isBias = True K automatically gets 1 added to it
    but the user does not need to append a bias value to input_u. User must also supply the training output, aka the teacher
    data. '''
    def train(self, input_u, teacher, washout):        
        time = np.shape(teacher)[0]                                         #time steps
        
        if input_u is None and not(self.isBias):                            #for feedback only reservoir
            input_u = np.zeros((time, self.K))
        elif input_u is None and self.isBias:
            input_u = np.ones((time, self.K))
        elif self.isBias:                                                   #if isBias then append a column of ones to input_u
            input_u = np.concatenate((input_u, np.ones((time,1))), axis=1)
        
        r = np.zeros((time-washout, self.L))
        x = (np.zeros((self.N))).reshape(-1,1)
        y = (np.zeros((self.L))).reshape(-1,1)
        
        #send off to appropriate train function based on selections
        if self.isClassification:
            self.trainClassification(time, input_u, teacher, r, x, y, washout)
        else:    
            if not(self.isU2Y) and not(self.isY2Y):
                self.trainBasic(time, input_u, teacher, r, x, y, washout)
            elif self.isU2Y and not(self.isY2Y):
                self.trainU2Y(time, input_u, teacher, r, x, y, washout)
            elif self.isU2Y and self.isY2Y:
                self.trainUY2Y(time, input_u, teacher, r, x, y, washout)
            elif not(self.isU2Y) and self.isY2Y:
                self.trainY2Y(time, input_u, teacher, r, x, y, washout)
            
        
        #Calculate Wout
        if self.outAlg == 0:                                                #moore-pseudo inverse == 0
            self.Wout = np.transpose((np.linalg.pinv(self.M)).dot(r))
        elif self.outAlg == 1:                                              #ridge regression == 1
            firstTerm =(np.transpose(self.T)).dot(self.M)
            secondTerm = np.linalg.inv( (np.transpose(self.M)).dot(self.M) + self.B * np.identity(self.M.shape[1]) )
            self.Wout = firstTerm.dot(secondTerm)
            
        return self.M, self.Wout #return state matrix and forced teacher output for each state
    
 #  _____               __  __      _   _               _     
 # |  __ \             |  \/  |    | | | |             | |    
 # | |__) |   _ _ __   | \  / | ___| |_| |__   ___   __| |___ 
 # |  _  / | | | '_ \  | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
 # | | \ \ |_| | | | | | |  | |  __/ |_| | | | (_) | (_| \__ \
 # |_|  \_\__,_|_| |_| |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
                                                            
                                                            
    '''No transforms on the reservoir state, input and previous output are not used in calculating prediction'''
    def runBasic(self, time, input_u, x, y, washout):
        resFunc = self.resFunc                                           #reservoir activation function
        outFunc = self.outFunc 
        outputs = np.zeros((time-washout, self.L))
        np.seterr(all='raise')
        try:
            for t in range(0,time):
                    u = (input_u[t]).reshape(-1,1)
                    WdotX = (self.W).dot(x)
                    WinDotU = (self.Win).dot(u)
                    WfbDotY = (self.Wfb).dot(y)
                    innerTerm = WdotX + WinDotU + WfbDotY + (self.sv*self.v[t]).reshape(-1,1)
                    theTanTerm = resFunc(innerTerm)
                    secondTerm = self.a * theTanTerm
                    x = (1 - self.a) * x + secondTerm
                    y = outFunc(((self.Wout).dot(x)).reshape(-1,1))
                    if t >= washout:
                        outputs[t-washout,:] = y.reshape(-1,self.L)
        except FloatingPointError:
            print('Exceptionally bad generation of ESN. Aborting sub-trial. (1)')
            output[:,:] = np.nan()
        return outputs
    
    '''No transforms on the reservoir state, input and previous output are not used in calculating prediction'''
    def runClassification(self, time, input_u, x, y, washout):
        resFunc = self.resFunc                                           #reservoir activation function
        outFunc = self.outFunc 
        outputs = np.zeros((time-washout, self.L))
        for t in range(0,time):
                u = (input_u[t]).reshape(-1,1)
                WdotX = (self.W).dot(x)
                WinDotU = (self.Win).dot(u)
                WfbDotY = (self.Wfb).dot(y)
                innerTerm = WdotX + WinDotU + WfbDotY + (self.sv*self.v[t]).reshape(-1,1)
                theTanTerm = resFunc(innerTerm)
                secondTerm = self.a * theTanTerm
                x = (1 - self.a) * x + secondTerm
                y = outFunc(((self.Wout).dot(x)).reshape(-1,1))
                if t >= washout:
                    outputs[t-washout,:] = y.reshape(-1,self.L)
        return outputs
    
    '''No transform in reservoir state, input is connected to output units'''
    def runU2Y(self, time, input_u, x, y, washout):
        resFunc = self.resFunc                                           #reservoir activation function
        outFunc = self.outFunc                                           #output activation funcion
        outputs = np.zeros((time-washout, self.L))
        for t in range(0,time):
                u = (input_u[t]).reshape(-1,1)
                WdotX = (self.W).dot(x)
                WinDotU = (self.Win).dot(u)
                WfbDotY = (self.Wfb).dot(y)
                innerTerm = WdotX + WinDotU + WfbDotY + (self.sv*self.v[t]).reshape(-1,1)
                theTanTerm = resFunc(innerTerm)
                secondTerm = self.a * theTanTerm
                x = (1 - self.a) * x + secondTerm
                ux = np.concatenate((u, x), axis=0)
                y = outFunc(((self.Wout).dot(ux)).reshape(-1,1))
                if t >= washout:
                    outputs[t-washout,:] = y.reshape(-1,self.L)
        return outputs
    
    '''input is connected to output units and there are self recurrent connections into the output unit'''
    def runUY2Y(self, time, input_u, x, y, washout):
        resFunc = self.resFunc                                           #reservoir activation function
        outFunc = self.outFunc                                           #output activation funcion
        outputs = np.zeros((time-washout, self.L))
        np.seterr(all='raise')
        try:
            for t in range(0,time):
                    u = (input_u[t]).reshape(-1,1)
                    WdotX = (self.W).dot(x)
                    WinDotU = (self.Win).dot(u)
                    WfbDotY = (self.Wfb).dot(y)
                    innerTerm = WdotX + WinDotU + WfbDotY + (self.sv*self.v[t]).reshape(-1,1)
                    theTanTerm = resFunc(innerTerm)
                    secondTerm = self.a * theTanTerm
                    x = (1 - self.a) * x + secondTerm
                    uxy = np.concatenate((u, x, y), axis=0)
                    y = outFunc(((self.Wout).dot(uxy)).reshape(-1,1))
                    if t >= washout:
                        outputs[t-washout,:] = y.reshape(-1,self.L)
        except FloatingPointError:
            print('Exceptionally bad generation of ESN. Aborting sub-trial. (1)')
            outputs[:,:] = np.nan
        return outputs
    
    '''self recurrent connections into the output unit'''
    def runY2Y(self, time, input_u, x, y, washout):
        resFunc = self.resFunc                                           #reservoir activation function
        outFunc = self.outFunc                                           #output activation funcion
        outputs = np.zeros((time-washout, self.L))
        for t in range(0,time):
                u = (input_u[t]).reshape(-1,1)
                WdotX = (self.W).dot(x)
                WinDotU = (self.Win).dot(u)
                WfbDotY = (self.Wfb).dot(y)
                innerTerm = WdotX + WinDotU + WfbDotY + (self.sv*self.v[t]).reshape(-1,1)
                theTanTerm = resFunc(innerTerm)
                secondTerm = self.a * theTanTerm
                x = (1 - self.a) * x + secondTerm
                xy = np.concatenate((x,y), axis=0)
                y = outFunc(((self.Wout).dot(xy)).reshape(-1,1))
                if t >= washout:
                    outputs[t-washout,:] = y.reshape(-1,self.L)
        return outputs
    
    '''Call this function with ESN object to get a prediction
    Precondition: ESN objected has been instatiated and train has been called. input_u to make predictions
    must be supplied unless this is a feedback driven reservoir, then specify None. time to run the ESN for must also be provided.
    washout and state are not required. Washout defaults to 0 and state defaults to the last reservoir state from 
    the previous train/run.
    Postcondition: Returns predictions'''
    def run(self, input_u, time, washout = 0, state = None):
        if input_u is None and not(self.isBias):                            #for feedback only reservoir
            input_u = np.zeros((time, self.K))
        elif input_u is None and self.isBias:
            input_u = np.ones((time, self.K))
        elif self.isBias:                                                   #if isBias then append a column of ones to input_u
            input_u = np.concatenate((input_u, np.ones((time,1))), axis=1)
        
        #start state
        if state is None: #Take the last state from training
            x = (self.M[-1,:self.N]).reshape(-1,1)
            y = (self.T[-1,:self.L]).reshape(-1,1)
        else:
            x = state.reshape(-1,1)
            y = (np.zeros((1,self.L))).reshape(-1,1) 
            #assuming for any chosen state to use previous Y state as zeros this may need changed

        #send off to appropriate run function based on selections
        outputs = None
                #send off to appropriate train function based on selections
        if self.isClassification:
            self.runClassification(time, input_u, x, y, washout)
        else:    
            if not(self.isU2Y) and not(self.isY2Y):
                outputs = self.runBasic(time, input_u, x, y, washout)
            elif self.isU2Y and not(self.isY2Y):
                outputs = self.runU2Y(time, input_u, x, y, washout)
            elif self.isU2Y and self.isY2Y:
                outputs = self.runUY2Y(time, input_u, x, y, washout)
            elif not(self.isU2Y) and self.isY2Y:
                outputs = self.runY2Y(time, input_u, x, y, washout)
                
        return outputs

