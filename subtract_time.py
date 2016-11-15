# -*- coding: utf-8 -*-
"""
Created on Tue Nov 06

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
import brute_with_cooling_heating_corr as brute

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


temps = lim['Temperature'].values
demands = lim['Demand'].values
demands = np.array(demands,dtype=float)

period_averages = np.zeros((48))
for i in range(48):
    period_averages[i] = np.mean(demands[i::48])
    
'''    
order = 10
def poly(x,*args):
    result = 0
    for i in range(order):
        result += args[i]*x**i
    return result
    
result = opt.curve_fit(poly,range(48),period_averages,p0=[0]*order)

fitted = [poly(i,result[0]) for i in range(48)]
'''

adjusted_demands = copy.deepcopy(demands)

for i in range(len(demands)):
    index = i%48
    adjusted_demands[i] -= period_averages[index]

'''    
plt.figure()
plt.plot(range(len(demands)),demands,c='g',label='real')
plt.plot(range(len(demands)),adjusted_demands,c='r',label='adjusted')
plt.legend()
'''

def new_optimise(args,demands=adjusted_demands,temps=temps):
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
    storage,heating = brute.sub_new(temps,storage,heating,Tb,K,A,Hb,C)
    result = 1-abs(np.corrcoef(heating,demands)[0][1])
    print args[0:2],result
    if math.isnan(result):
        return np.inf
    return result

result = opt.basinhopping(new_optimise,[100,-100,1,1,0.01,20,0.01],
                          stepsize=.5,niter=200)
print result                         