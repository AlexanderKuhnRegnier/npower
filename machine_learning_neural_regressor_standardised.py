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
import copy

def power_prep(lim,powers):
    frames = []
    for power in powers:
        new = lim**power
        new.columns = [column+'_power_{:}'.format(power) for column in new.columns]
        frames.append(new)
    return pd.concat(frames,axis=1)

def previous_days_prep(lim,days):
    frames = []
    for day in days:
        new = lim.copy().shift(day)
        new.columns = [column+'_shift_{:}'.format(day) for column in new.columns]
        frames.append(new)
    final = pd.concat(frames,axis=1)
    final = final.iloc[max(days):]
    return final

#plt.close('all')

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
#lim['Index'] = lim.index.values    #Include index numbers as a dataset (not shown to be overly great)
#lim = lim[[column for column in lim.columns if column not in ['Precipitation Type','Wind Direction','Cloud Cover']]]
orig = copy.deepcopy(lim)
N = 45000
lim = lim.iloc[:N]
rest = orig.iloc[N:]

data = lim[[column for column in lim.columns if 'Demand' not in column]]
days = range(0,48*8,12)
#days = [0]
dataframe = previous_days_prep(data,days)
columns = dataframe.columns.tolist()
data = dataframe.values

rest_data = rest[[column for column in rest.columns if 'Demand' not in column]]
rest_data = previous_days_prep(rest_data,days).values

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)
rest_data = scaler.transform(rest_data)

demands = np.array(lim['Demand'].values,dtype=np.float64)[max(days):]
demands_future = np.array(rest['Demand'].values,dtype=np.float64)[max(days):]
from sklearn.neural_network import MLPRegressor

training_file = 'training2'
if os.path.isfile(training_file):
    with open(training_file,'rb') as f:
        neural = pickle.load(f)
else:
    print "starting new..."
    neural = MLPRegressor(verbose=True,max_iter=10000,hidden_layer_sizes=(100,),tol=1e-10,activation='logistic',
                          learning_rate='adaptive')
    print neural.fit(data,demands)
    with open(training_file,'wb') as f:
        pickle.dump(neural,f,protocol=2)        
        
score = neural.score(data,demands)
print "Score: ",score
prediction = neural.predict(data)

plt.figure()
plt.plot(range(len(prediction)),demands,c='b',label='real')
plt.plot(range(len(prediction)),neural.predict(data),c='r',label='predicted')
plt.legend(loc=1)
plt.title('score: '+str(score))

plt.figure()
plt.plot(range(len(demands_future)),demands_future,c='b',label='real')
plt.plot(range(len(demands_future)),neural.predict(rest_data),c='r',label='predicted')
plt.legend(loc=1)
plt.title('Predicting '+'score: '+str(neural.score(rest_data,demands_future)))