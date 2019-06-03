# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:07:45 2019
https://github.com/AndrewCarterUK/pascal-voc-writer
@author: hfuji
"""

import os
import glob
from pascal_voc_writer import Writer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

classes = ["raccoon"]
target_dir = 'D:/Develop/data/raccoon_dataset/'

files = glob.glob(target_dir + 'Labels/*.txt')

for fpath in files:
    basename = os.path.basename(fpath)
    img_path = target_dir + 'JPEGImages/' + basename[:-4] + '.jpg'
    if not os.path.exists(img_path):
        continue
    img = mpimg.imread(img_path) #image to array
    height, width = img.shape[:2]
    plt.imshow(img) #array to 2Dfigure
    plt.show()
    with open(fpath, 'rt') as fp:
        lines = fp.readlines()
        
        writer = Writer(img_path, width, height)
        for line in lines:
            line = line.strip()
            words = line.split(' ')
            label = int(words[0])
            x = int(float(words[1]) * width)
            y = int(float(words[2]) * height)
            w = int(float(words[3]) * width)
            h = int(float(words[4]) * height)
            xmin = int(x - w / 2)
            ymin = int(y - h / 2)
            xmax = int(x + w / 2)
            ymax = int(y + h / 2)
            writer.addObject(classes[label], xmin, ymin, xmax, ymax)
        
        out_path = target_dir + 'tmp/' + basename[:-4] + '.xml'
        writer.save(out_path)
       
#    if len(lines) > 1:
#        print(lines)
#        break
