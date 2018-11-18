# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:57:21 2018
https://webbibouroku.com/Blog/Article/sklearn-cluster-flag
https://qiita.com/deaikei/items/11a10fde5bb47a2cf2c2
http://download.mvtec.com/halcon-10.0-solution-guide-ii-d-classification.pdf
https://github.com/amueller/introduction_to_ml_with_python/blob/master/03-unsupervised-learning.ipynb
@author: hfuji
"""

import os
import glob
from PIL import Image
import numpy as np
from skimage import data
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
#import random
import matplotlib.pyplot as plt

in_dir = 'input/'

files = glob.glob(in_dir + '*.png')
label_map = ['c', 'thumb', 'fist', 'palm', 'ok']
label = []
for fpath in files:
    img = Image.open(fpath)
    basename = os.path.basename(fpath)
    words = basename.split('_')
    label.append(words[1])
#random.shuffle(files)

# 画像データを２次元配列に変換
feature = np.array([data.imread(f'{path}') for path in files])
feature = feature.reshape(len(feature), -1).astype(np.float64)

# ５グループにクラスタリング
#model = KMeans(n_clusters=5).fit(feature)
#labels = model.labels_
km = KMeans(n_clusters=5, random_state=0)
km.fit(feature)

# 分類結果表示
y_km = km.predict(feature)
print(y_km)
print(label)
print("ARI: {:.2f}".format(adjusted_rand_score(y_km, label)))

# 距離特徴量
distance_feature = km.transform(feature)
print(distance_feature)
#x = np.linspace(0, 4, 5)
for idx in range(len(files)):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(label_map, distance_feature[idx], marker='o')
#    ax.set_xlim(0, 4)
    ax.set_ylim(0, 30000)
    ax.set_xlabel("gesture type")
    ax.set_ylabel("distance")
#    plt.close(fig)

'''
# クラスタ中心の画像化
for idx in range(5):
    img_cent = km.cluster_centers_[idx].reshape(200,200).astype(np.uint8)
    plt.imshow(img_cent, cmap='gray')
    plt.show()
'''