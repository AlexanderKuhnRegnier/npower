# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 18:04:45 2016

@author: ahk114
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import os
from numpy.fft import fft
from numpy.fft import fftfreq
import scipy.optimize as opt
import copy
from scipy.optimize import minimize as minimize
import math
import time
from numba import jit,float64
from matplotlib.widgets import Slider, Button, RadioButtons
from sklearn.preprocessing import StandardScaler
import data_prep
from sklearn.neural_network import MLPRegressor

def scale(values):
    scaler = StandardScaler()
    scaler.fit(values)
    result = scaler.transform(values)
    return result,scaler

def select_columns(frame,to_select):
    if type(to_select) == str:
        to_select = [to_select]
    return frame[[column for column in frame.columns if column in to_select]]   

def exclude_columns(frame,to_exclude):
    if type(to_exclude) == str:
        to_exclude = [to_exclude]
    return frame[[column for column in frame.columns if column not in to_exclude]]   

def source(sourcefile,fill_with_mean,ignore_unknown_rows=True):
    '''
    returns data,demand,data_demand --- all dataframes, last one is 
                                        concatenation of the first two
    '''
    if fill_with_mean:
        ignore_unknown_rows = True
    else:
        print ('''Unknown rows included, returning a frame which may contain
                object type columns, manual conversion may be required!''')
        ignore_unknown_rows = False
    '''
    load data and demand dataframes
    '''
    data = data_prep.load_data(source=sourcefile,fill_with_mean=fill_with_mean,
                               ignore_unknown_rows=ignore_unknown_rows)
    demand = data_prep.load_demand(source=sourcefile,fill_with_mean=fill_with_mean,
                               ignore_unknown_rows=ignore_unknown_rows)
    '''
    Join up the data and the demand columns into one dataframe
    '''
    data_demand = pd.concat((data,demand),axis=1)
    return data,demand,data_demand


def NN(sourcefile,to_discard,fill_with_mean,ignore_unknown_rows,
                  periods,MLP_kwargs):
    '''
    inputs:
        +source input data file
        +limit term for number of data points DISCARDED from this input data
        +options for data parising - ie. filling with mean or not, or ignoring 
            unknown rows
        +the periods for the data to be shifted for
        +additional arguments for the MLPRegressor
    returns:
        +neural - the MLPRegressor instance
        +score - score for the training data
        +actual demand values - used for training (with terms at the end and 
                                                beginning chopped off due to
                                                day shifts (beginning) and 
                                                potentially the limiting term)
    '''    
    data,demand,data_demand = source(sourcefile=sourcefile,
                                     fill_with_mean=fill_with_mean,
                                     ignore_unknown_rows=ignore_unknown_rows)
    data = data.iloc[:len(data)-to_discard]
    demand = demand.iloc[:len(demand)-to_discard]
    data_demand = data_demand.iloc[:len(data_demand)-to_discard]
    N = len(demand) #number of rows, ie. number of data points
    '''
    Number of periods to be shifted
    '''
    assert 0 in periods, "There should be a non-shifted set!"
    '''
    Shift the data and demand columns by the specified amount (in periods list)
    '''
    shifted_data_demand = data_prep.previous_days_prep(data_demand,periods)
    shifted_N = len(shifted_data_demand)    #number of entries after shift
    
    actual_demand = select_columns(shifted_data_demand,['Demand_shift_0'])
    #below still contains shifted data though!
    shifted_data = exclude_columns(shifted_data_demand,['Demand_shift_0'])
    
    shifted_data_vals = shifted_data.values
    actual_demand_vals = actual_demand.values
    
    shifted_data_vals,shifted_data_scaler = scale(shifted_data_vals)
    
    '''
    scale the demand as well, if needed - would need to be scaled back afterwards,
    has not been shown to do much, so commented out for now
    '''
    #actual_demand_scaler = StandardScaler()
    #actual_demand_scaler.fit(actual_demand_vals)
    #actual_demand_vals = actual_demand_scaler.transform(actual_demand_vals)
    '''
    #specify additional options if needed - now in function arg as MLP_kwargs
    neural = MLPRegressor(verbose=True,max_iter=10000,hidden_layer_sizes=sizes,tol=1e-10,activation=activation,
                          learning_rate=learning_rate,random_state=np.random.randint(0,100),
                          beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,
                          learning_rate_init=learning_rate_init,alpha=alpha)
    '''
    neural = MLPRegressor(**MLP_kwargs)
    print neural.fit(shifted_data_vals,actual_demand_vals)   #perform the fitting
    score = neural.score(shifted_data_vals,actual_demand_vals)
    print "score with data from training:",score
    print "corrcoef for same:",np.corrcoef(neural.predict(shifted_data_vals).reshape(-1,),
                                           actual_demand_vals.reshape(-1,))[0][1]
    return neural,score,actual_demand_vals                                           
                                           
MLP_kwargs = {
'verbose':True,
'max_iter':10000,
'hidden_layer_sizes':(100,),
'tol':1e-10,
'activation':'logistic',
}             

neural,score,training_demand_vals=NN(sourcefile='round_2.xlsx',
                                     to_discard = 0,
                                     fill_with_mean=True,
                                     ignore_unknown_rows=True,
                                     periods = range(0,10),
                                     MLP_kwargs = MLP_kwargs)