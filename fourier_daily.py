# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 18:46:11 2016

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

#plt.close('all')

data = data_prep.load_data(source='round_1.xlsx',fill_with_mean=True)
demand = data_prep.load_demand(source='round_1.xlsx',fill_with_mean=True).values
temperature = data['Temperature'].values
N = len(temperature)
def lin(x,A,B):
    return A*x+B
    
DX,DC = opt.curve_fit(lin,range(N),demand)[0]
TX,TC = opt.curve_fit(lin,range(N),temperature)[0]

adjusted_demand = [i - lin(i,DX,DC) for i in demand]
adjusted_temperature = [i - lin(i,TX,TC) for i in temperature]

'''
plt.figure()
plt.title('adjusted things')
plt.plot(range(N),adjusted_demand,label='demand')
plt.plot(range(N),adjusted_temperature,label='temp')
plt.legend()
plt.show()
'''

daily_demand_fourier = fft(adjusted_demand)*(1./N)
daily_temperature_fourier = fft(adjusted_temperature)*(1./N)
assert daily_demand_fourier.size == daily_temperature_fourier.size, 'Sizes should match!'

freqs = fftfreq(daily_temperature_fourier.size,d=1)
print "Number of points:",daily_demand_fourier.shape

'''
plt.figure()
plt.title('Fourier components')
plt.scatter(freqs,daily_temperature_fourier,label='temperature')
plt.scatter(freqs,daily_demand_fourier,label='demand',c='r')
plt.legend()
plt.show()
'''

temperature_fourier = pd.DataFrame({'frequency':freqs,'amplitude':np.abs(daily_temperature_fourier),
                                    'phase':np.angle(daily_temperature_fourier)})
                                    
demand_fourier = pd.DataFrame({'frequency':freqs,'amplitude':np.abs(daily_demand_fourier),
                                    'phase':np.angle(daily_demand_fourier)})
                                    
limit = 5000

temperature_fourier=temperature_fourier.iloc[:limit]
demand_fourier=demand_fourier.iloc[:limit]

temperature_fourier.sort_values(by='amplitude',ascending=False,inplace=True)                                    
demand_fourier.sort_values(by='amplitude',ascending=False,inplace=True)

print "temp fourier"
print temperature_fourier

print "demand fourier"
print demand_fourier

more_points = 10000

def add_fourier_components(fourier_frame,N=N):
    output = np.zeros(N+more_points)
    for i,k in fourier_frame.iterrows():
        amplitude = k['amplitude']
        phase = k['phase']
        frequency = k['frequency']
        output += amplitude*np.cos((2*np.pi*frequency*np.arange(N+more_points))+phase)
    return output
    
demand_components = add_fourier_components(demand_fourier)
temperature_components = add_fourier_components(temperature_fourier)

data2 = data_prep.load_data(source='round_2.xlsx',fill_with_mean=True)
demand2 = data_prep.load_demand(source='round_2.xlsx',fill_with_mean=True).values
temperature2 = data2['Temperature'].values
adjusted_demand2 = [i - lin(i,DX,DC) for i in demand2]
adjusted_temperature2 = [i - lin(i,TX,TC) for i in temperature2]


t = range(N+more_points)
plt.figure()
plt.title('%d components added together' % limit)
plt.plot(range(len(adjusted_demand2)),adjusted_demand2,c='r',label='round2 demand')
plt.plot(t,demand_components,c='b',label='round 1 predicted demand')
plt.legend()
plt.show()
plt.figure()
plt.title('%d components added together' % limit)
plt.plot(range(len(adjusted_demand2)),adjusted_temperature2,c='m',label='round2 temp')
plt.plot(t,temperature_components,c='g',label='round 1 predicted temp')
plt.legend()
plt.show()