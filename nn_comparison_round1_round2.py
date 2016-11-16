# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:22:46 2016

@author: ahk114
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import data_prep
from sklearn import preprocessing
import cPickle as pickle

forecast = pd.read_excel('forecast.xlsx')
round2 = pd.read_excel('round_2.xlsx')
round2lim = round2[round2['Date'] >= pd.Timestamp('2014-10-01')]
round2lim = round2lim[round2lim['Date'] <= pd.Timestamp('2015-03-31')]
demandforecast = forecast['Demand'].values
demandround2lim = round2lim['Demand'].values
demandround2lim = np.asarray(demandround2lim,dtype=float)
print np.corrcoef(demandforecast,demandround2lim)
plt.figure()
plt.plot(range(len(demandround2lim)),demandround2lim,c='b',label='round2')
plt.plot(range(len(demandround2lim)),demandforecast,c='r',label='forecast')
plt.legend()
plt.title('comparing our forecast to the actual data')
plt.show()

round1_data = data_prep.load_data(ignore_unknown_rows=False,fill_with_mean=True)

days = [0, 8, 16, 24, 32, 40, 48, 56]

round1_data_shifted = data_prep.previous_days_prep(round1_data,days)
#columns = dataframe.columns.tolist()
round1_data_shifted = round1_data_shifted.values

#rest_data = rest[[column for column in rest.columns if 'Demand' not in column]]
#rest_data = previous_days_prep(rest_data,days).values
scaler = preprocessing.StandardScaler().fit(round1_data_shifted)
round1_data_shifted = scaler.transform(round1_data_shifted)
#rest_data = scaler.transform(rest_data)

with open('neural/neural.pickle','rb') as f:
    neural = pickle.load(f)
    
round1_data_prediction = neural.predict(round1_data_shifted)

plt.figure()
plt.plot(range(len(demandround2lim)),demandround2lim,c='b',label='round2')
plt.plot(range(len(demandround2lim)),round1_data_prediction[-len(demandround2lim):],c='r',label='forecast with all data')
plt.legend()
plt.title('comparing full data forecast to the actual data')
plt.show()

np.corrcoef(demandround2lim,round1_data_prediction[-len(demandround2lim):])