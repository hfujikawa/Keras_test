# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 10:48:52 2019

@author: hfuji
"""

import pandas as pd
import matplotlib.pyplot as plt

fpath = 'D:/Develop/Samples/ssd_keras/optuna_ssd_raccoon.csv'
'''
df = pd.read_csv(fpath)
df = df[['trial_id', 'state']]
df = df.drop([df.index[0]])
print(df)
'''

# https://own-search-and-study.xyz/2016/10/19/pandas%E3%81%A7header%E3%81%8C%E8%A4%87%E6%95%B0%E8%A1%8C%E3%81%82%E3%82%8B%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%82%92%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%82%80%E6%96%B9%E6%B3%95/
#2行だけデータ読み込み
temp = pd.read_csv(fpath,
                     header=None,
                     nrows=2)
#Header作成
temp2 = temp.fillna(' ')
header = temp2.ix[0] + "[" + temp2.ix[1] + "]"
 
#作成結果確認
print(header)

#3行目以降のデータだけ読み込み
df = pd.read_csv(fpath,
                     header=None,
                     skiprows=3)
df = df.drop([0], axis=1)
#結果確認
print(df)

#カラム名にheaderを指定
df.columns = header[1:]
 
#結果確認
print(df)

df = df.ix[:,8:]
print(df)

# https://stackoverflow.com/question,s/32105817/plot-entire-row-on-pandas
row = df.iloc[1]
#ind = range(row.shape[0])
row.plot()
#plt.plot()
plt.show()
#    row = df.iloc[i]
#    print(len(row))
#    row.plot()
#plt.show()
