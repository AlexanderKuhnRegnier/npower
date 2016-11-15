# -*- coding: utf-8 -*-
"""
Created on Tue Nov 03

@author: Make Weather Models Great Again
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import cPickle as pickle
import os
from numpy.fft import fft
from numpy.fft import fftfreq
import scipy.optimize as opt
import copy
from scipy.optimize import minimize as minimize
import math
import time
from numba import jit

plt.close('all')

pd.options.display.expand_frame_repr = False
pd.options.display.max_columns = 15
source = 'round_1.xlsx'
picklefile = 'round1.pickle'
if not os.path.isfile(picklefile):
    df = pd.read_excel(source)
    with open(picklefile,'wb') as f:
        pickle.dump(df,f,protocol=2)
else:
    with open(picklefile,'rb') as f:
        df = pickle.load(f)
    
print "Rows, columns:", df.shape

lim = df.loc[df['Demand'] != '?? FC1 ??']
new_columns = [i for i in lim.columns if 'Unnamed' not in i]
lim = lim[new_columns]
lim = lim.fillna(lim.mean())
'''
Heat_at_i = Heat_at_(i-1) + (baseline - Temp*Time) 
start at i = 0 and H(0) = baseline-timp*time
'''
outputs = []
lim_copy = copy.deepcopy(lim)
lim = lim[[column for column in lim.columns if 'Date' not in column]]
lim = lim.rolling(window=48).mean()[47::48]

temps = lim['Temperature'].values
demands = lim['Demand'].values
demands = np.array(demands,dtype=float)


@jit(nopython=True)
def baseline(array,Temperature,index,Tb,K,A,Hb,C):
    heating = A*(Hb-array[index-1])
    if heating >= 0:
        result = array[index-1] + (Temperature-Tb)*K + heating
    else:
        result = array[index-1] + (Temperature-Tb)*K - C*(Hb-array[index-1])
        heating = 0
    return result,heating

@jit(nopython=True)
def sub_new(temps,storage,heating,Tb,K,A,Hb,C):    
    for i in range(1,len(temps)):
        storage[i],heating[i] = baseline(storage,temps[i],i,Tb,K,A,Hb,C)
    return storage,heating
    
def new_optimise(args,demands=demands,temps=temps):
    H =args[0]
    S=args[1]
    Tb = args[2]
    K = args[3]
    A = args[4]
    Hb = args[5]
    C = args[6]
    storage = np.zeros(len(temps))
    heating = np.zeros(len(temps))
    heating[0] = H
    storage[0] = S
    storage,heating = sub_new(temps,storage,heating,Tb,K,A,Hb,C)
    result = 1-abs(np.corrcoef(heating,demands)[0][1])
    print args[0:2],result
    if math.isnan(result):
        return np.inf
    return result

start = time.clock()
result = opt.basinhopping(new_optimise,[150,-200,17,10,0.3,355,0.001],stepsize=2.5,niter=100)
'''
result = opt.brute(new_optimise,ranges=((150,300),
                                        (-200,200),
                                        (16,21),
                                        (0.1,10),
                                        (0.001,1),
                                        (0,400),
                                        (0.001,0.1)),
                    Ns = 10,full_output=True,finish=None)
'''
print "finished"
print result
#print result[0]
#print result[1]
duration = time.clock()-start
print "duration:",duration,duration/60.

def linear(x,a,b):
    return a*x + b

def plotting(args,demands=demands,temps=temps):
    H =args[0]
    S=args[1]
    Tb = args[2]
    K = args[3]
    A = args[4]
    Hb = args[5]
    C = args[6]
    storage = np.zeros(len(temps))
    heating = np.zeros(len(temps))
    heating[0] = H
    storage[0] = S
    storage,heating = sub_new(temps,storage,heating,Tb,K,A,Hb,C)
    plt.figure()
    plt.hexbin(demands,heating)
    plt.show()
    plt.figure()
    plt.scatter(demands,heating)
    plt.show()
    plt.figure()
    straight = opt.curve_fit(linear,heating,demands)[0]
    heating_adjusted = heating*straight[0] + straight[1]
    plt.plot(range(len(lim_copy)),lim_copy['Demand'].values,c='g',label='demands')
    plt.plot(np.array(range(len(heating)))*48,heating_adjusted[:],c='r',label='heating')
    plt.legend()
    plt.figure()
    heating_adjusted = heating*straight[0] + straight[1]
    plt.plot(range(len(heating)),demands,c='g',label='demands')
    plt.plot(np.array(range(len(heating))),heating_adjusted[:],c='r',label='heating')    
plotting(result.x)