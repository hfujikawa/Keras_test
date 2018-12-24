# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 08:36:44 2018

@author: hfuji
"""

import pandas as pd
import matplotlib.pyplot as plt

fpath = 'ssd300_raccoon_training_from_h5_cnv4_log.csv'
df = pd.read_csv(fpath)
print(df)

epoch = df['epoch'].values
loss = df['loss'].values
val_loss = df['val_loss'].values

plt.plot(epoch, loss, 'b-')
plt.plot(epoch, val_loss, 'r-')
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.show()
