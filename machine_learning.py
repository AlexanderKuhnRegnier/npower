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

def linear(lim,plot=True):
    from sklearn import linear_model
    target = lim['Demand'].values
    data = lim[[column for column in lim.columns if column != 'Demand']]
    print "target shape:",target.shape
    print "data shape:",data.shape
    reg = linear_model.LinearRegression()
    reg.fit(data.values,target.reshape(-1,1))
    columns = data.columns.tolist()
    coefficients = reg.coef_[0].tolist()
    print coefficients
    print columns
    display_frame = pd.DataFrame({'column':columns,'coefficient':coefficients})
    print display_frame
    display_frame = pd.DataFrame({'column':columns,
                    'coefficient':np.array(coefficients)/data.mean().values})
    print display_frame
    data = data.values
    predicted = [reg.predict(sample.reshape(1,-1))[0][0] for sample in data]
    x = range(len(predicted))
    if plot:
        plt.figure(figsize=(22,12))
        plt.plot(x,target,c='g',label='real')
        plt.plot(x,predicted,c='r',label='predicted')
        plt.legend()
        plt.title('Plotting predicted vs real values with linear model')
        plt.show()
    return reg

def prepped_linear(target,data,plot=True):
    from sklearn import linear_model
    data = data[[column for column in data.columns if 'Demand' not in column]]
    print "target shape:",target.shape
    print "data shape:",data.shape
    reg = linear_model.LinearRegression()
    reg.fit(data.values,target.reshape(-1,1))
    columns = data.columns.tolist()
    coefficients = reg.coef_[0].tolist()
    print coefficients
    print columns
    display_frame = pd.DataFrame({'column':columns,'coefficient':coefficients})
    print display_frame
    print "sorted"
    print display_frame.sort_values('coefficient',ascending=False).iloc[:10]
    display_frame = pd.DataFrame({'column':columns,
                    'coefficient':np.array(coefficients)/data.mean().values})
    print display_frame
    data = data.values
    predicted = [reg.predict(sample.reshape(1,-1))[0][0] for sample in data]
    x = range(len(predicted))
    if plot:
        plt.figure(figsize=(22,12))
        plt.plot(x,target,c='g',label='real')
        plt.plot(x,predicted,c='r',label='predicted')
        plt.legend()
        plt.title('Plotting predicted vs real values with linear model')
        plt.show()
    return reg
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
        new = lim.copy().shift(day)
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

print "Filtered rows,columns:", lim.shape

#linear(lim)
shifts = [0,1,2,3]
target = lim['Demand'].iloc[max(shifts):]
powered = power_prep(lim,[1,2,3,4,5])
shifted = previous_days_prep(powered,shifts)
prepped_linear(target,shifted)