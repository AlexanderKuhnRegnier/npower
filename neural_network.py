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
#import copy
from scipy.optimize import minimize as minimize
#import math
#import time
#from numba import jit,float64
#from matplotlib.widgets import Slider, Button, RadioButtons
from sklearn.preprocessing import StandardScaler
import data_prep
from sklearn.neural_network import MLPRegressor

#%%
def predict_row_by_row(neural,scaler,sourcefile,
                       start_chop=56924,periods=range(3),data_points=100):
    '''
    need at least max(period)-1 actual data points for this to work!
    '''
    assert 0 in periods, "There should be a non-shifted set!"
    data = data_prep.load_data(source=sourcefile,fill_with_mean=True,
                               ignore_unknown_rows=False)
    demand = data_prep.load_demand(source=sourcefile,fill_with_mean=False,
                               ignore_unknown_rows=False)
    print data.shape,demand.shape
    #demand = demand.iloc[start_chop:len(demand)-end_chop]
    #data = data.iloc[start_chop:len(data)-end_chop]     
    demand = demand.iloc[start_chop:start_chop+data_points]
    data = data.iloc[start_chop:start_chop+data_points]
    print data.shape,demand.shape                 
    '''
    Data is ready to be processed, demand still has to be modified
    '''
    shifted_data = data_prep.previous_days_prep(data,periods)
    '''
    Processing the demand - replace unknown values with 0, then shift 
    and join up
    '''
    #print np.sum(demand.isnull())
    def replace(x):
        if type(x) == str or type(x) == unicode:
        #print x
            replace.counter += 1
            return 0
        else:
            return x
    replace.counter = 0
    #print type(demand)
    demand['Demand'] = demand['Demand'].apply(replace)
    #print np.sum(demand.isnull())
    #print type(demand),demand.columns
    demand['Demand'] = pd.to_numeric(demand['Demand'])
    #print demand
    #print demand[demand['Demand']>0]
    
    #print "unknown:",replace.counter
    #print type(demand)
    '''
    original index:
        0
        1
        2
        3
        4
        5
        .
        .
        .
    if periods = [0,1,2,3]
    then index would look like:
        3
        4
        5
        .
        .
        .
    ie 0,1,2 cut off the beginning -> max_period-1 diminished
    this will always be true, no matter how long the original index is
    
    so at iteration 0 (the first one), from above example, you would have 
    data from row 3, -> estimate of demand at row 3.
    Add this estimate to row 3 in the original list of demands,
    and then repeat this process
    '''
    max_shift = max(periods)
    for i in range(data.shape[0]-max(periods)): #go through all shifted data
        shifted_demand = data_prep.previous_days_prep(demand,periods)
        #print "shifted demand"
        #print shifted_demand
        shifted_demand_data = exclude_columns(shifted_demand,['Demand_shift_0'])
        #print "shifted data 0"
        #print shifted_data.iloc[i].values
        #print shifted_demand_data.iloc[i].values
        input_data = np.append(shifted_data.iloc[i].values,shifted_demand_data.iloc[i].values).reshape(1,-1)
        #print "input_data",input_data
        scaled_input_data = scaler.transform(input_data)
        #print "scaled input data"
        #print scaled_input_data
        estimate = neural.predict(scaled_input_data)
        #print "estimated",estimate
        demand.iloc[max_shift+i] = estimate
        #print "new demand"
        #print demand
        #demand.ix[i+max_shift,'Demand'] = estimate
        #print "new demand"
        #print demand
    return demand
#%%
        
        
def predict_from_shifted_data(neural,data,demands):
    predicted = neural.predict(data)
    score = neural.score(data,demands)
    print "Prediction score:",score
    return score,predicted

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
    #data_demand = pd.concat((data,demand),axis=1)
    return data,demand

def NN_prep(sourcefile,start_chop,end_chop,fill_with_mean,
            ignore_unknown_rows,periods):
    data,demand = source(sourcefile=sourcefile,
                                     fill_with_mean=fill_with_mean,
                                     ignore_unknown_rows=ignore_unknown_rows)
    data = data.iloc[start_chop:len(data)-end_chop]
    demand = demand.iloc[start_chop:len(demand)-end_chop]
    #data_demand = data_demand.iloc[start_chop:len(data_demand)-end_chop]
    #N = len(demand) #number of rows, ie. number of data points
    '''
    Number of periods to be shifted
    '''
    assert 0 in periods, "There should be a non-shifted set!"
    '''
    Shift the data and demand columns by the specified amount (in periods list)
    '''
    shifted_data = data_prep.previous_days_prep(data,periods)
    shifted_demand = data_prep.previous_days_prep(demand,periods)
    #shifted_N = len(shifted_data_demand)    #number of entries after shift
    
    actual_demand = select_columns(shifted_demand,['Demand_shift_0'])
    #below still contains shifted data though!
    shifted_demand = exclude_columns(shifted_demand,['Demand_shift_0'])  
    shifted_data = pd.concat((shifted_data,shifted_demand),axis=1)
    return shifted_data,actual_demand
    
def NN(sourcefile,end_chop,fill_with_mean,ignore_unknown_rows,
                  periods,MLP_kwargs,neural_in=None):
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
        +scaler - used to scale the input data (except orig. demand)
        +score - score for the training data
        +actual demand values - used for training (with terms at the end and 
                                                beginning chopped off due to
                                                day shifts (beginning) and 
                                                potentially the limiting term)
        +predicted demand values - same shape as above
    '''    
    shifted_data,actual_demand = NN_prep(sourcefile=sourcefile,
                                         start_chop=0,
                                         end_chop=end_chop,
                                         fill_with_mean=fill_with_mean,
                                         ignore_unknown_rows=ignore_unknown_rows,
                                         periods=periods)
    
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
    if neural_in:
        print """Using passed in MLPRegressor instance, make sure the parameters
                are right!"""
        neural = neural_in
    else:
        neural = MLPRegressor(**MLP_kwargs)
    print neural.fit(shifted_data_vals,actual_demand_vals)   #perform the fitting
    score = neural.score(shifted_data_vals,actual_demand_vals)
    print "score with data from training:",score
    predicted = neural.predict(shifted_data_vals).reshape(-1,)
    print "corrcoef for same:",np.corrcoef(predicted,
                                           actual_demand_vals.reshape(-1,))[0][1]
    return neural,shifted_data_scaler,score,actual_demand_vals,predicted                                     

    
    
'''
options for NN set here, as a dictionary
'''      
periods = range(3)                                   
MLP_kwargs = {
'verbose':True,
'max_iter':10000,
'hidden_layer_sizes':(1500,),
'tol':1e-10,
'activation':'logistic',
'warm_start':True,
'max_iter':10000,
}             
'''
further options set here
'''
total_N = data_prep.load_data('round_2.xlsx').shape[0]
end_chop = 1000
neural,shifted_data_scaler,score,training_demand_vals,predicted_demand_vals=NN(
                                    sourcefile='round_2.xlsx',
                                    end_chop = end_chop,
                                    fill_with_mean=True,
                                    ignore_unknown_rows=True,
                                    periods = periods,
                                    MLP_kwargs = MLP_kwargs)


predicted_demand = predict_row_by_row(neural,shifted_data_scaler,'round_2.xlsx',
                                      start_chop = total_N-end_chop-max(periods),
                                        data_points = 800,
                                        periods=periods)
                                                                                
real = data_prep.load_demand('round_2.xlsx')
real2 = real.iloc[total_N-end_chop-max(periods):total_N-end_chop-max(periods)+800]
t= range(len(real2))   
plt.figure()                                                      
plt.plot(t,real2.values,c='b',label='real')           
plt.plot(t,predicted_demand.values,c='r',label='predicted')            
plt.legend()
plt.show()