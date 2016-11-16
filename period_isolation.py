# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:14:42 2016

@author: ahk114
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import os
from numpy.fft import fft
from numpy.fft import fftfreq
import scipy.optimize as opt
import copy
from scipy.optimize import minimize as minimize
import math
import time
from numba import jit,float64
from matplotlib.widgets import Slider, Button, RadioButtons
from sklearn.preprocessing import StandardScaler
import data_prep

data = data_prep.load_data(fill_with_mean=True)
demand = data_prep.load_demand(fill_with_mean=True)
N = len(demand)
period_group1 = range(14,31)

period_group2 = [i for i in range(0,48) if i not in period_group1]
select = []
counter = 0

def selection(*args):
    global selection,counter
    
    index = counter
    remainder = index%48
    
    if remainder in period_group1:
        select.append(args[0])
    counter += 1
    
demand.apply(selection)