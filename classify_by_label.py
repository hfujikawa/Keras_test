# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 05:45:18 2019

@author: hfuji
"""

import os
import glob
import shutil

label_dir = 'D:/Develop/data/raccoon_dataset/Labels/'
img_dir = 'D:/Develop/data/raccoon_dataset/JPEGImages/'
dst0_dir = 'D:/Temp/'
dst1_dir = ''

files = glob.glob(label_dir + '*.txt')

for idx, fpath in enumerate(files):
    if idx > 2:
        break
    with open(fpath, 'rt') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            words = line.split(' ')
        label = int(words[0])
    
    basename = os.path.basename(fpath)
    basename = basename[:-4] + '.jpg'
    src_path = img_dir + basename
    if label == 0:
        dst_path = dst0_dir + basename
    shutil.copyfile(src_path, dst_path)

