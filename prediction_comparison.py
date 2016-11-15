# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:22:46 2016

@author: ahk114
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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