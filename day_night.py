# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 17:08:37 2016

@author: ahk114
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

temps = df['Temperature']

days = []
nights = []
for i,temp in enumerate(temps):
    print "i",i
    if i%48<12 or i%48>=36:
        nights.append(temp)
    else:
        days.append(temp)
day_avg = []
night_avg = []
for i in range(0,len(days),24):
    print "i2"
    day_avg.append(np.mean(days[i:i+24]))
for i in range(0,len(nights),24):
    print "i3",i
    night_avg.append(np.mean(nights[i:i+24]))

dnight = []
dday = []
for i,demand in enumerate(df['Demand'].loc[df['Demand'] != '?? FC1 ??']):
    print "i",i
    if i%48<12 or i%48>=36:
        dnight.append(demand)
    else:
        dday.append(demand)
demand_day_avg = []
demand_night_avg = []
for i in range(0,len(dday),24):
    print "i2"
    demand_day_avg.append(np.mean(dday[i:i+24]))
for i in range(0,len(dnight),24):
    print "i3",i
    demand_night_avg.append(np.mean(dnight[i:i+24]))
    

plt.figure() 
plt.scatter(day_avg[:len(demand_day_avg)],demand_day_avg,c='g',label='day')
plt.scatter(night_avg[:len(demand_night_avg)],demand_night_avg,c='b',label='night')
plt.legend()
plt.xlabel('temperature')
plt.ylabel('demand')
