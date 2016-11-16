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

data = data_prep.load_data(source='round_2.xlsx',fill_with_mean=True)
demand = data_prep.load_demand(source='round_2.xlsx',fill_with_mean=True)
N = len(demand)

data_demand = pd.concat((data,demand),axis=1)

days = range(0,50)
assert 0 in days, "There should be a non-shifted set!"
shifted_data_demand = data_prep.previous_days_prep(data_demand,days)
shifted_N = len(shifted_data_demand)

actual_demand = shifted_data_demand[['Demand_shift_0']]
shifted_data_demand = shifted_data_demand[[column for column in 
            shifted_data_demand.columns if column not in ['Demand_shift_0']]]

shifted_data_demand_vals = shifted_data_demand.values
actual_demand_vals = actual_demand.values

shifted_data_demand_scaler = StandardScaler()
shifted_data_demand_scaler.fit(shifted_data_demand_vals)
shifted_data_demand_vals = shifted_data_demand_scaler.transform(shifted_data_demand_vals)

#actual_demand_scaler = StandardScaler()
#actual_demand_scaler.fit(actual_demand_vals)
#actual_demand_vals = actual_demand_scaler.transform(actual_demand_vals)
'''
neural = MLPRegressor(verbose=True,max_iter=10000,hidden_layer_sizes=sizes,tol=1e-10,activation=activation,
                      learning_rate=learning_rate,random_state=np.random.randint(0,100),
                      beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,
                      learning_rate_init=learning_rate_init,alpha=alpha)
'''
neural = MLPRegressor(verbose=True,max_iter=10000,hidden_layer_sizes=(500,),
                      tol=1e-10)
print neural.fit(shifted_data_demand_vals,actual_demand_vals)
    
score = neural.score(shifted_data_demand_vals,actual_demand_vals)
print score