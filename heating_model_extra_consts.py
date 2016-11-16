# -*- coding: utf-8 -*-
"""
Created on Tue Nov 03

@author: Make Weather Models Great Again
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
    
print "Raw Dataset, rows, columns:", df.shape
lim = df.loc[df['Demand'] != '?? FC1 ??']
new_columns = [i for i in lim.columns if 'Unnamed' not in i]
lim = lim[new_columns]

lim = lim.fillna(lim.mean())
    
temps = lim['Temperature'].values
demands = lim['Demand'].values
demands = np.array(demands,dtype=float)

@jit(nopython=True)
def model(const,temps,predictions,index):
    Tb = const[0]*(1+const[1]*np.sin((2*np.pi*index/(365.25*48))+const[2]))
    T = temps[index]
    remainder = index%48
    alpha = const[5+remainder]
    '''
    if Tb<T:
        heating = (1+alpha)*(Tb-T)*const[3]
    else:
        heating = 0
    '''
    '''
    #only last half hour, 2 half hours, 3 half hours
    if index>2:
        retention = (const[11]*predictions[index-1]+
                    const[12]*predictions[index-2]+
                    const[13]*predictions[index-3])
    else:
        retention = 0
    '''
    if index>48*3:
        retention = (
                    const[55]*np.mean(predictions[index-48:index])+
                    const[56]*np.mean(predictions[index-(48*2):index-48])+
                    const[57]*np.mean(predictions[index-(48*3):index-(48*2)])
                    )
    else:
        retention = 0
    heating = (1+alpha)*(Tb-T)*const[3] + retention
    loss = const[4]*(Tb-T)
    if index%48<14 or index%48>=42:
        #night constants
        return heating+loss+const[53]
    else:
        #day heating
        return heating+loss+const[54]

@jit(nopython=True)
def sub_new(const,temps,predictions):    
    for i in range(0,len(temps)):
        predictions[i] = model(const,temps,predictions,i)
    return predictions
    
def new_optimise(const,demands=demands,temps=temps):
    predictions = np.zeros(len(temps))
    predicted = sub_new(const,temps,predictions)
    result = 1-abs(np.corrcoef(predicted,demands)[0][1])
    print (', '.join(['%+.2e']*const.shape[0])) % tuple(const),result
    if math.isnan(result):
        return 1e100
    return result
    
def get_results(const,demands=demands,temps=temps):
    predictions = np.zeros(len(temps))
    #storage = np.zeros(len(temps+3))
    #storage[:3] = np.array([0,0,0])
    print "Calling subnew"
    predicted = sub_new(const,temps,predictions)
    result = 1-abs(np.corrcoef(predicted,demands)[0][1])
    print (', '.join(['%+.2e']*const.shape[0])) % tuple(const),result
    if math.isnan(result):
        print "ERROR"
        return 1e100,np.zeros((len(demands)))
    return result,predicted

to_insert = ((-1.,1.),)

bounds = (
(0,40),
(-0.4,.4),  
(0,2*np.pi),
(-100,200),
(-100,200))

to_insert = to_insert*48
bounds += to_insert
bounds += ((-400,400),
            (-400,400),
            (-1,1),
            (-1,1),
            (-1,1)
            )
constmin = [bound[0] for bound in bounds]
constmax = [bound[1] for bound in bounds]

#const = np.array([40,0.239,1.49,75.7,2.22,-1.02,-1.01,-1,-0.989,-27.7,23.6,-0.655,0.414,0.212],dtype=np.float64)
'''
const = np.array([ 39.89759094,   0.23986434,   1.48782018,  79.82501341, 
                  1.95616592,  -1.02203147,  -1.02042863,  -1.01845583,   
                  -1.01585489,  -6.45209124,   5.06102748,  -0.66028467,  
                  0.42286959,   0.20653274],dtype=np.float64)
'''

const = np.array([0]*len(bounds),dtype=float)

basinhopping = 1

if basinhopping:
    class MyBounds(object):
         def __init__(self, xmax, xmin):
             self.xmax = np.array(xmax)
             self.xmin = np.array(xmin)
         def __call__(self, **kwargs):
             x = kwargs["x_new"]
             tmax = bool(np.all(x <= self.xmax))
             tmin = bool(np.all(x >= self.xmin))
             return tmax and tmin
    
    def print_fun(x, f, accepted):
        print("at minimum %.4f accepted %d" % (f, int(accepted)))
    
    mybounds = MyBounds(constmax,constmin)
    
    result = opt.basinhopping(new_optimise,const,niter=10,
                              callback=print_fun,accept_test=mybounds)
'''
x = []
f = []
for i in range(1):
    x0 = [np.random.uniform(bounds[k][0],bounds[k][1]) for k in range(len(bounds))]
    x.append(x0)
    result = opt.minimize(new_optimise,x0,method='TNC',
                          bounds = bounds,options={'disp':False})
    f.append(result.fun)
    print "finished:",('[ '+', '.join(['%+.2e']*len(x0))+' ] ') % tuple(x0),result.fun
'''
'''
0 Tb
1 A
2 phi
3 Tp
4 Lp
5         4 alphas, for 4 equally divided times of the day
6
7
8
'''
'''
ranges = (
slice(0,40,10),
slice(-0.1,.1,0.02),  
slice(0,np.pi,0.4),
slice(0,200,35),
slice(0,200,35),
slice(-1,1,1),
slice(-1,1,1),
slice(-1,1,1),
slice(-1,1,1)
)

result = opt.brute(new_optimise,ranges=ranges,finish=None,full_output=True)
'''
'''
fit = get_results(const)[1]
    
scaler = StandardScaler()
scaler.fit(demands.reshape(-1,1))
fitscaler = StandardScaler()
fitscaler.fit(fit.reshape(-1,1))

scaled_demands = scaler.transform(demands.reshape(-1,1))
scaled_fit = fitscaler.transform(fit.reshape(-1,1))

adjusted_demands = scaled_demands - scaled_fit

itransformed_data = scaler.inverse_transform(adjusted_demands.reshape(-1,1))

plt.figure()
plt.plot(range(len(itransformed_data)),itransformed_data,c='r',label='adjusted')
plt.legend()
plt.show()
'''