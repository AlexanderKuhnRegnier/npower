import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

match = 'nn_results'
result_files = []
for f in os.listdir(os.getcwd()):
    if match in f:
        result_files.append(f)

lines = []        
for fi in result_files:
    with open(fi,'r') as f:
        lines.extend(f.readlines())
        
lines = [line.replace('\n','') for line in lines]
nns = lines[0::2]
results = lines[1::2]

string_indices = [2,3]

split = []
split = [r.split('|') for r in results]
for i in range(len(split)):
    for j in range(len(split[i])):
        if j in string_indices:
            split[i][j] = split[i][j]
        else:
            split[i][j] = eval(split[i][j])

split.sort(reverse=True,key=lambda x:x[-1])

columns = [[l[i] for l in split] for i in range(len(split[0]))]

names = ['period_shifts_upper','period_shifts_stepsize',
         'hidden_layers_number','hidden_layer_mean_size','hidden_layers_std',
         'activation','learning_rate',
         'beta_1','beta_2','epsilon','learning_rate_init',
         'alpha','test_score','real_score']

data=[]
         
for i in range(len(columns)):
    if i==0:
       data.append([max(l) for l in columns[i]])
       data.append([l[1] for l in columns[i]])
    elif i==1:
        data.append([len(l) for l in columns[i]])
        data.append([np.mean(l) for l in columns[i]])
        data.append([np.std(l) for l in columns[i]])
    else:
        data.append(columns[i])
        
data_dict = dict([(key,value) for key,value in zip(names,data)])

df = pd.DataFrame(data_dict)

filtered = df[df['real_score']>0.6]