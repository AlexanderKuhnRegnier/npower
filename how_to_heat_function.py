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
        result = array[index-1] + (Tb-Temperature)*K + A*(Hb-array[index-1])
    else:
        result = array[index-1] + (Tb-Temperature)*K
    return result

def baseline_old(array,Temperature,index,Tb,K,A,Hb):
    global counter    
    result = array[index-1] + (Tb-Temperature)*K + A*(Hb-array[index-1])
    '''    
    print "output:",result
    counter += 1
    prit counter
    '''
    #if result>1e4:
        #raise Exception("Something went fucking wrong, large value:{:}".format([array[index-1],array[index-2],[array[index+i] for i in np.arange(0,-20,-1)],Temperature,index,Tb,K,A,Hb]))
    #if math.isnan(result):
        #raise Exception("Something went fucking wrong:{:}".format([array[index-1],array[index-2],[array[index+i] for i in np.arange(0,-20,-1)],Temperature,index,Tb,K,A,Hb]))
    #outputs.append(result)
    if math.isnan(result):
        raise ValueError
    return result
    
temps = lim['Temperature'].values
demands = lim['Demand'].values
demands = np.array(demands,dtype=float)
final_result = 0

def optimise(S,Tb,K,A,Hb,demands=demands,temps=temps):
    #global final_result
    #print S,Tb,K,A,Hb
    storage = np.zeros(len(temps))
    storage[0] = S
    for i in range(1,len(temps)):
        try:
            storage[i] = baseline(storage,temps[i],i,Tb,K,A,Hb)
        except ValueError:
            return 1e300
    final_result = 1-abs(np.corrcoef(storage,demands)[0][1])
    return final_result
    
def optimise_return_storage(S,Tb,K,A,Hb,demands=demands,temps=temps):
    print S,Tb,K,A,Hb
    storage = np.zeros(len(temps))
    storage[0] = S
    for i in range(1,len(temps)):
        try:
            storage[i] = baseline(storage,temps[i],i,Tb,K,A,Hb)
        except ValueError:
            return 1e300
    return 1-abs(np.corrcoef(storage,demands)[0][1]),storage

def optimise_wrapper(*args):
    return optimise(*args[0])

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
    print args,result
    return result

constants = 5
start =time.clock()
steps = 3000

#result = opt.basinhopping(new_optimise,[50,1,8,.003,6],stepsize=0.6,niter=5000)
result = opt.brute(new_optimise,ranges=((0,100),
                                        (0,10),
                                        (0,10),
                                        (0,1),
                                        (0,10)),
                    Ns = 2)
print "finished"
print result
raise StopIteration('STOOOOOP')
R = np.random.uniform

data = np.zeros((steps,constants+1))
for i in range(steps):
    duration = time.clock()-start
    avg_duration = duration/(i+1)
    print "Time remaining {:.3f} s {:.1f} m".format((steps-i-1)*avg_duration,(steps-i-1)*avg_duration/60.)
    print "{:}%".format(i*100./steps)    
    x0=[R(-70,70) for blabla in range(constants)]
    data[i,0:-1] = x0
    result = optimise(*x0)
    print result    
    data[i,-1] = result
   
final = data[data[:,-1].argsort()]
mask = np.abs(final[:,-1])<1
final = final[mask]
selection = final[:10]
datafile = 'data2'
print "final:", final
if os.path.isfile(datafile):
    with open(datafile,'rb') as f:
        old = pickle.load(f)
else:
    old = []
with open(datafile,'wb') as f:
    old.extend(selection.tolist())
    pickle.dump(old,f,protocol=2)
    
def read_data(filename=datafile):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    print data
    return data