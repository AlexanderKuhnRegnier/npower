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

@jit(nopython=True)
def baseline(array,Temperature,index,Tb,K,A,Hb):
    if A*(Hb-array[index-1]) >= 0:  
        result = array[index-1] + (Temperature-Tb)*K + A*(Hb-array[index-1])
    else:
        result = array[index-1] + (Temperature-Tb)*K
    return result
   
temps = lim['Temperature'].values
demands = lim['Demand'].values
demands = np.array(demands,dtype=float)
final_result = 0

@jit(nopython=True)
def sub_new(temps,storage,Tb,K,A,Hb):    
    for i in range(1,len(temps)):
        storage[i] = baseline(storage,temps[i],i,Tb,K,A,Hb)
    return storage
    
def new_optimise(args,demands=demands,temps=temps):
    S=args[0]
    Tb = args[1]
    K = args[2]
    A = args[3]
    Hb = args[4]
    storage = np.zeros(len(temps))
    storage[0] = S
    storage = sub_new(temps,storage,Tb,K,A,Hb)
    result = 1-abs(np.corrcoef(storage,demands)[0][1])
    if math.isnan(result):
        return np.inf
    print args,result
    return result

constants = 5
start =time.clock()
steps = 3000

#result = opt.basinhopping(new_optimise,[50,1,8,.003,6],stepsize=0.6,niter=5000)
result = opt.brute(new_optimise,ranges=((0,100),
                                        (0,30),
                                        (0,20),
                                        (0,3),
                                        (0,20)),
                    Ns = 20,full_output=True,finish=None)
print "finished"
print result
print result[0]
print result[1]