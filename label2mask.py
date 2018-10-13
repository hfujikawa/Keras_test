# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 20:41:15 2018

@author: hfuji
"""

import os
from PIL import Image
import glob

src_dir = 'D:/Develop/data/mapillary/input/training/labels/'
src_dir = 'D:/Develop/data/mapillary/input/validation/labels/'

files = glob.glob(src_dir + '*.png')
cnt = 0
train_list = []
for file_path in files:
    full_path = os.path.join(src_dir, file_path)
    img = Image.open(full_path)
    colors = img.convert('RGB').getcolors() #this converts the mode to RGB
    
    for color in colors:
#        print(color[0], color[1])
        if color[1] == (0,0,80):
            cnt += 1
            basename = os.path.basename(file_path)
            print(cnt, basename[:-4], color[0], color[1])
            train_list.append(basename)

with open('val.txt','w') as f:
    f.write( '\n'.join(train_list) )