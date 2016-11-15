# -*- coding: utf-8 -*-
"""
Created on Tue Nov 03 2016

@author: Make Weather Models Great Again
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import cPickle as pickle
import numpy as np

def power_prep(lim,powers):
    lim = lim.copy()
    frames = []
    for power in powers:
        new = lim**power
        new.columns = [column+'_power_{:}'.format(power) for column in new.columns]
        frames.append(new)
    return pd.concat(frames,axis=1)

def previous_days_prep(lim,days):
    lim = lim.copy()
    frames = []
    for day in days:
        new = lim.shift(day)
        new.columns = [column+'_shift_{:}'.format(day) for column in new.columns]
        frames.append(new)
    final = pd.concat(frames,axis=1)
    final = final.iloc[max(days):]
    return final



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
'''
new_columns = [i for i in lim.columns if 'Unnamed' not in i and 'Period' not in i
                    and 'Date' not in i]
'''  
new_columns = [i for i in lim.columns if 'Unnamed' not in i
                    and 'Date' not in i]                  
lim = lim[new_columns]
lim = lim.fillna(lim.mean())

demands = np.array(lim['Demand'].values,dtype=np.float64)
data = np.array(lim[[column for column in lim.columns if 'Demand' not in column]].values,dtype=np.float64)

print "Filtered rows,columns:", lim.shape

from sklearn import svm

clf = svm.SVR()
print "starting..."
print clf.fit(data[:2000],demands[:2000])

prediction = [clf.predict(row.reshape(1,-1)) for row in data]

plt.figure()
plt.plot(range(len(prediction)),demands[:len(prediction)],c='b',label='real')
plt.plot(range(len(prediction)),prediction,c='r',label='predicted')
plt.legend(loc=1)