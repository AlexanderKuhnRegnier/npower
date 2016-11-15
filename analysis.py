# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 18:17:23 2016

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

new_columns = [i for i in lim.columns if 'Unnamed' not in i]
lim = lim[new_columns]

TOTALSCATTER=0

if TOTALSCATTER:
    to_plot = [i for i in lim.columns if 'Demand' not in i]
    fig = plt.figure()
    for i,column in zip(range(10),to_plot):
        ax = fig.add_subplot(2,5,i+1)
        ax.scatter(lim[column].values,lim['Demand'].values)
        ax.set_xlabel(column)
        ax.set_ylabel('Demand')
 
HEXBIN = 1
if HEXBIN:
    to_plot = [i for i in lim.columns.tolist() if 'Demand' not in i]
    for i,column in zip(range(10),to_plot):
        plt.figure()
        plt.title(column)
        plt.hexbin(lim[column].values,lim['Demand'].values)
        plt.colorbar()
        #ax.set_xlabel(column)
        #ax.set_ylabel('Demand')       

DAYS = 0
if DAYS:
    days = lim.groupby('Date')
    means = days.mean()
    
    fourier = fft(means - np.mean(means))
    fourier = fourier*(1./fourier.shape[0])
    freqs = fftfreq(fourier.size,d=1)
    
    plt.figure()
    plt.scatter(freqs,fourier)
    plt.ylabel('fourier stuff')
    plt.title('Grouped by day')
    
    plt.figure()
    plt.scatter(range(means.shape[0]),means.Temperature)
    plt.ylabel('average temps')

'''
Replace nan values with mean of the entire column!

For FFT, need to divide by N!!!!!!!
'''
new_df = lim.fillna(lim.mean())

FOURIER_FILLED = 0
if FOURIER_FILLED:
    data = new_df['Temperature']
    
    #fourier = fft(data - np.mean(data),norm='ortho')
    #fourier = fft(data,norm='ortho')
    fourier = fft(data)
    fourier = fourier*(1./fourier.shape[0])
    freqs = fftfreq(fourier.size,d=1)
    
    print "Number of points:",fourier.shape
    
    plt.figure()
    plt.scatter(freqs,fourier)
    plt.ylabel('fourier stuff')
    plt.title('all points')
    
    fourierframe = pd.DataFrame({'f':freqs,'a':np.abs(fourier)})
    fourierframe.sort_values('a',inplace=True)
    print fourierframe
    
    plt.figure()
    plt.scatter(range(means.shape[0]),means.Temperature)
    plt.ylabel('temps')
    plt.title('Entire Data')

'''
Subtract linear component from data, 
then fourier transform again
Limit data points, select central data showing 
triangle wave pattern only
'''

FILTERED_FOURIER_TEMP=0

filtered = new_df.iloc[1600:54000]

def lin(x,intercept,gradient):
    return x*gradient + intercept
    
popt,pcov = opt.curve_fit(lin,range(filtered.shape[0]),filtered.Temperature)

intercept = popt[0]
grad = popt[1]

filtered_s = copy.deepcopy(filtered)
filtered_s.loc[:,'Temperature'] = filtered_s.loc[:,'Temperature']-intercept
lin_points = [lin(x,0,grad) for x in range(filtered_s.shape[0])]
filtered_s.loc[:,'Temperature'] = filtered_s.Temperature - lin_points

data = filtered_s.Temperature.values
fourier = fft(data)
fourier = fourier*(1./fourier.shape[0])
freqs = fftfreq(fourier.size,d=1)

n = len(freqs)
n = int(n/2)-1
fourierframe = pd.DataFrame({'frequency':freqs[:n],'amplitude':np.abs(fourier[:n]),'phase':np.angle(fourier[:n])})
fourierframe.sort_values('amplitude',inplace=True,ascending=False)
print fourierframe.iloc[:15]

plt.figure()
plt.scatter(fourierframe['frequency'],fourierframe['amplitude'])
plt.ylabel('fourier stuff')
plt.title('all points, linear adjusted')

'''
Compare results to actual data!
'''

def func_lin(t,amplitude,frequency,phase,intercept=intercept,grad=grad):
    return amplitude*np.cos((2*np.pi*t*frequency)+phase)+lin(t,intercept,grad)

def func(t,amplitude,frequency,phase):
    return amplitude*np.cos((2*np.pi*t*frequency)+phase)
    
'''
take first elements from fourierframe
3 different forms, take 1st, 1st and 2nd, 1st, 2nd and 3rd, etc..
'''

'''
plot each on their own!
'''

#plt.figure()
#colours = ['r','m','g']
r1 = np.zeros(filtered.shape[0])
for i in range(5):
    row = fourierframe.iloc[i]
    amplitude = row['amplitude']
    phase = row['phase']
    frequency = row['frequency']
    colour = 'g'
    N = filtered.shape[0]
    #plt.subplot(1,3,i+1)
    #plt.figure()
    print "amplitude, frequency, phase",amplitude,frequency,phase
    new = [func(t,amplitude,frequency,phase) for t in xrange(N)]
    r1 += new
    #plt.scatter(range(N),new,c=colour,label=str(i+1))
    #plt.legend(loc='best')
plt.figure()
plt.scatter(range(N),filtered['Temperature'],c='r',label='real')
plt.scatter(range(N),r1+np.array([lin(t,intercept,grad) for t in range(N)]),label='all together')
plt.legend()
'''
plt.figure()
colours = ['r','m','g']
for i,colour in zip(range(2,3),colours):
    rows = fourierframe.iloc[:i+1]
    amplitudes = rows['amplitude'].values
    frequencies = rows['frequency'].values
    phases = rows['phase'].values
    N = filtered.shape[0]
    results = np.empty(N)
    for amplitude,frequency,phase in zip(amplitudes,frequencies,phases):
        new = [func(t,amplitude,frequency,phase) for t in xrange(N)]        
        results += new
    plt.scatter(range(N),results,c=colour,label=str(i+1))
#np.array([lin(t,intercept,grad) for t in range(N)])    
plt.scatter(range(N),filtered['Temperature'].values,label='real temp')
plt.legend()
'''


GROUP_DAYS = 1

mean_temps = []
N = filtered.shape[0]
step = 30*48
for i in range(0,N-step,step):
    mean_temps.append(filtered['Temperature'].iloc[i:i+step].mean())
mean_temps = np.array(mean_temps)
plt.figure()
plt.scatter(range(len(mean_temps)),mean_temps,c='r',label='mean over %i days' % (step/48.))  
plt.legend()

def lin(x,intercept,gradient):
    return x*gradient + intercept
    
popt,pcov = opt.curve_fit(lin,range(mean_temps.shape[0]),mean_temps)

intercept = popt[0]
grad = popt[1]

mean_temps = mean_temps-intercept
lin_points = [lin(x,0,grad) for x in range(mean_temps.shape[0])]
mean_temps = mean_temps - lin_points

data = mean_temps
#fourier = fft(data - np.mean(data),norm='ortho')
#fourier = fft(data,norm='ortho')
fourier = fft(data)
fourier = fourier*(1./fourier.shape[0])
freqs = fftfreq(fourier.size,d=1)

print "Number of points:",fourier.shape

plt.figure()
plt.scatter(freqs,np.abs(fourier))
plt.ylabel('fourier transform of monthly temp average')

fourierframe2 = pd.DataFrame({'frequency':freqs,'amplitude':np.abs(fourier),'phase':np.angle(fourier)})
fourierframe2.sort_values('amplitude',inplace=True,ascending=False)
print fourierframe2.iloc[:15]

ratio = 100.
r1 = np.zeros(len(mean_temps)*ratio)
N=r1.shape[0]
for i in range(5):
    row = fourierframe2.iloc[i]
    amplitude = row['amplitude']
    phase = row['phase']
    frequency = row['frequency']
    colour = 'g'
    #plt.subplot(1,3,i+1)
    #plt.figure()
    print "amplitude, frequency, phase",amplitude,frequency,phase
    new = [func(t,amplitude,frequency,phase) for t in np.array(range(N))/ratio]
    r1 += new
    #plt.scatter(range(N),new,c=colour,label=str(i+1))
    #plt.legend(loc='best')
plt.figure()
plt.scatter(range(len(mean_temps)),mean_temps,c='r',label='real')
#plt.scatter(np.array(range(N))/ratio,r1+np.array([lin(t,intercept,grad) for t in range(N)]),label='all together')
plt.scatter(np.array(range(N))/ratio,r1,label='all together')
plt.legend()
plt.title('monthly averages')
