# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:09:42 2018
P.102 deep-learning-with-python-notebooks
@author: hfuji
"""

#from keras.datasets import boston_housing
import numpy as np

fpath = "D:/Develop/data/raccoon_dataset/data/train.txt"
data = []
with open(fpath) as fp:
    lines = fp.readlines()
    for line in lines:
        data.append(line.strip())

k = 5
num_val = len(data) // k
np.random.shuffle(data)
validation_scores = []

for fold in range(k):
    validation_data = data[num_val * fold:num_val * (fold+1)]
    training_data = data[:num_val * fold] + data[num_val * (fold+1):]
    
    with open('val{}.txt'.format(fold),'w') as f:
        f.write( '\n'.join(validation_data) )
    with open('train{}.txt'.format(fold),'w') as f:
        f.write( '\n'.join(training_data) )
