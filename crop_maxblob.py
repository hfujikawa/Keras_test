# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:57:39 2018
https://www.kaggle.com/gti-upm/leaphandgestuav
@author: hfuji
"""

import cv2
import glob
import os
import matplotlib.pyplot as plt

W = 200
H = 200
img_dir = 'D:/Develop/data/leapGestRecog/00/01_palm/'
img_dir = 'D:/Develop/data/leapGestRecog/00/03_fist/'
img_dir = 'D:/Develop/data/leapGestRecog/00/05_thumb/'
img_dir = 'D:/Develop/data/leapGestRecog/00/07_ok/'
img_dir = 'D:/Develop/data/leapGestRecog/00/09_c/'
par_dir = os.path.basename(os.path.dirname(img_dir))
out_dir = 'output'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
files = glob.glob(img_dir + '*.png')

for idx, file_path in enumerate(files):
    if idx % 20 == 0:
#    if idx == 0:
        basename = os.path.basename(file_path)
        print(basename)
        img_gray = cv2.imread(file_path, 0)
        height, width = img_gray.shape[:2]
        ret, thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
        plt.imshow(thresh, cmap='gray')
        plt.show()
        _, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # select blob region and crop
        x_min = width
        y_min = height
        x_max = 0
        y_max = 0
        for cont in cnts:
            xmin, ymin, w, h = cv2.boundingRect(cont)
            xmax = xmin + w
            ymax = ymin + h
            print(xmin, ymin, xmax, ymax)
            x_min = min(xmin, x_min)
            y_min = min(ymin, y_min)
            x_max = max(xmax, x_max)
            y_max = max(ymax, y_max)
        img_crop = img_gray[y_min-5:y_max+5, x_min-5:x_max+5]
        img_resize = cv2.resize(img_crop, (W, H), interpolation=cv2.INTER_CUBIC)
        plt.imshow(img_resize, cmap='gray')
        plt.show()
        cv2.imwrite(out_dir + '/' + par_dir + basename[5:], img_resize)
                