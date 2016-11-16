# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:58:48 2016

@author: ahk114
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
def load_data(source = 'round_1.xlsx',
              ignore_unknown_rows = True,ignore_columns=['Unnamed','Date',
                                                         'Unnamed: 2'],
              fill_with_mean=False,convert_to_float=True):
    picklefile = source.replace('.xlsx','.pickle')
    if not os.path.isfile(picklefile):
        df = pd.read_excel(source)
        with open(picklefile,'wb') as f:
            pickle.dump(df,f,protocol=2)
    else:
        with open(picklefile,'rb') as f:
            df = pickle.load(f)
    print "Columns:", df.columns
    print "Loaded data: rows, columns = ", df.shape
    if ignore_unknown_rows:
        df = df.loc[(df['Demand'] != '?? FC1 ??') & (df['Demand'] != '?? FC2 ??') & (df['Demand'] != '?? FC3 ??')]

    new_columns = [i for i in df.columns if i not in ignore_columns+['Demand']]                  
    df = df[new_columns]
    #lim['Index'] = lim.index.values    #Include index numbers as a dataset (not shown to be overly great)
    #lim = lim[[column for column in lim.columns if column not in ['Precipitation Type','Wind Direction','Cloud Cover']]]
    if fill_with_mean:
        df = df.fillna(df.mean())
    if convert_to_float:
        df = df.apply(lambda x:np.array(x,dtype=float))
    return df
    
def load_demand(source = 'round_1.xlsx',
              ignore_unknown_rows = True,fill_with_mean=False,
              convert_to_float=True):
    picklefile = source.replace('.xlsx','.pickle')
    if not os.path.isfile(picklefile):
        df = pd.read_excel(source)
        with open(picklefile,'wb') as f:
            pickle.dump(df,f,protocol=2)
    else:
        with open(picklefile,'rb') as f:
            df = pickle.load(f)            
    print "Loaded data: rows, columns = ", df.shape
    if ignore_unknown_rows:
        df = df.loc[(df['Demand'] != '?? FC1 ??') & (df['Demand'] != '?? FC2 ??') & (df['Demand'] != '?? FC3 ??')]
    df = df['Demand']
    if fill_with_mean:
        df = df.fillna(df.mean())
    if convert_to_float:
        df = df.apply(lambda x:np.array(x,dtype=float))
    return df