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
import cPickle as pickle

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

data = lim[[column for column in lim.columns if 'Demand' not in column]]

#hidden_layers = np.random.randint(1,4)
#nr_days = np.random.randint(0,200)
#day_step = np.random.randint(int(nr_days/10.)+1,(int(nr_days/10.)+1)*2)

#activation = ['logistic','relu','tanh'][np.random.randint(0,2)]
activation = 'logistic'
#learning_rate = ['adaptive','constant','invscaling'][np.random.randint(0,2)]
learning_rate = 'constant'
#beta_1 = np.random.uniform(0.85,0.93)
#beta_2 = np.random.uniform(0.93,0.9999)
#epsilon = np.random.uniform(7e-9,13e-8)
#learning_rate_init = np.random.uniform(0.0005,0.0015)
#alpha = np.random.uniform(0.00007,0.00013)
beta_1 = 0.868184
beta_2 = 0.980315
epsilon=3.47961e-08
learning_rate_init =0.000588723
alpha = 7.29928e-05

'''
sizes = []
for i in range(hidden_layers):
    sizes.append(np.random.randint(30,1000))
sizes = tuple(sizes)
'''
#sizes = (np.random.randint(200,4000),)*hidden_layers
sizes = (1375,)

#days = range(0,12*nr_days,4*day_step)
#cutoff = np.random.randint(1,60)
#days = range(0,48*5,4)[0:cutoff:int(round(1.+(1/15.)*cutoff))]
days = [0, 8, 16, 24, 32, 40, 48, 56]

print "days, sizes:", days, sizes
dataframe = previous_days_prep(data,days)
#columns = dataframe.columns.tolist()
data = dataframe.values

#rest_data = rest[[column for column in rest.columns if 'Demand' not in column]]
#rest_data = previous_days_prep(rest_data,days).values
print "Finished shifting"
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)
#rest_data = scaler.transform(rest_data)

demands = np.array(lim['Demand'].values,dtype=np.float64)[max(days):]
#demands_future = np.array(rest['Demand'].values,dtype=np.float64)[max(days):]
from sklearn.neural_network import MLPRegressor
'''
neural = MLPRegressor(verbose=True,max_iter=10000,hidden_layer_sizes=sizes,tol=1e-10,activation=activation,
                      learning_rate=learning_rate,random_state=np.random.randint(0,100),
                      beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,
                      learning_rate_init=learning_rate_init,alpha=alpha)
'''
neural = MLPRegressor(verbose=True,max_iter=10000,hidden_layer_sizes=sizes,tol=1e-10,activation=activation,
                      learning_rate=learning_rate)    
print neural.fit(data,demands)
    
score = neural.score(data,demands)
#rest_score = neural.score(rest_data,demands_future)

consts = [days,sizes,activation,learning_rate,beta_1,beta_2,epsilon,learning_rate_init,alpha]

'''
plt.figure()
prediction = neural.predict(data)
plt.plot(range(len(prediction)),demands,c='b',label='real')
plt.plot(range(len(prediction)),neural.predict(data),c='r',label='predicted')
plt.legend(loc=1)
plt.title('score: '+str(score))

plt.figure()
plt.plot(range(len(demands_future)),demands_future,c='b',label='real')
plt.plot(range(len(demands_future)),neural.predict(rest_data),c='r',label='predicted')
plt.legend(loc=1)
plt.title('Predicting '+'score: '+str(neural.score(rest_data,demands_future)))
'''