# -*- coding: utf-8 -*-
"""
Created on Tue Nov 04

@author: Make Weather Models Great Again
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

'''
Neglect 'Period' column. - 10 other Columns
Plot Scatter plot of all other categories against Demand
'''
lim = df.loc[df['Demand'] != '?? FC1 ??']

new_columns = [i for i in lim.columns if 'Unnamed' not in i and 'Period' not in i]
lim = lim[new_columns]

TOTALSCATTER=0

new_df = lim.fillna(lim.mean())

data = new_df['Demand']
month_periods = 48*30
monthly = data.rolling(window=month_periods).mean()[month_periods:-1:month_periods]
data = monthly
#fourier = fft(data - np.mean(data),norm='ortho')
#fourier = fft(data,norm='ortho')
fourier = fft(data)
fourier = fourier*(1./fourier.shape[0])
freqs = fftfreq(fourier.size,d=1)

print "Number of points:",fourier.shape

plt.figure()
plt.scatter(freqs,np.abs(fourier))
plt.ylabel('fourier stuff')
plt.title('all points')

fourierframe = pd.DataFrame({'frequency':freqs,'amplitude':np.abs(fourier),'phase':np.angle(fourier)})
fourierframe.sort_values('amplitude',inplace=True,ascending=False)
demand_frame = copy.deepcopy(fourierframe)

print fourierframe

data = new_df['Temperature']
month_periods = 48*30
monthly = data.rolling(window=month_periods).mean()[month_periods:-1:month_periods]
data = monthly
#fourier = fft(data - np.mean(data),norm='ortho')
#fourier = fft(data,norm='ortho')
fourier = fft(data)
fourier = fourier*(1./fourier.shape[0])
freqs = fftfreq(fourier.size,d=1)

print "Number of points:",fourier.shape

#plt.figure()
plt.scatter(freqs,np.abs(fourier),c='r')
plt.ylabel('fourier stuff')
plt.title('all points')

fourierframe = pd.DataFrame({'frequency':freqs,'amplitude':np.abs(fourier),'phase':np.angle(fourier)})
fourierframe.sort_values('amplitude',inplace=True,ascending=False)
print fourierframe

frames = [fourierframe,demand_frame]
names = ['temperature','demand']
colours = ['r','g']
plt.figure()
N = 3
factor = 30.
for frame,name,colour in zip(frames,names,colours):
    filtered = frame.iloc[:N]
    y_list = []
    x = range(int(factor*fourierframe.shape[0]))
    for k,row in filtered.iterrows():
        amplitude = row['amplitude']
        frequency = row['frequency']
        phase = row['phase']
        y = [amplitude*np.cos((2*np.pi*frequency*i/factor)+phase) for i in x]
        y_list += y
    plt.scatter(x,y,c=colour,label=name)
plt.legend()
plt.show()