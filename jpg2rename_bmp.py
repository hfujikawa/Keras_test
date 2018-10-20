# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 07:48:47 2018

@author: hfuji
"""

import os
from PIL import Image
import glob
import shutil

src_jpg_dir = 'D:/Develop/data/VOCdevkit/VOC2007/JPEGImages/'
dst_bmp_dir = 'D:/Temp/'

jpg_files = glob.glob(src_jpg_dir + '*.jpg')

cnt = 0
for jpg_file in jpg_files:
    basename = os.path.basename(jpg_file)
    if int(basename[:-4]) % 10 == 0:
        cnt += 1
        dirname = os.path.dirname(jpg_file)
        dirs = dirname.split('/')
        new_fname = dirs[-2] + '_' + basename[:-4] + '.bmp'
        dst_bmp_path = dst_bmp_dir + new_fname
        print(dst_bmp_path)
#        pil_img = Image.open(jpg_file)
#        pil_img.save(dst_bmp_path, "bmp")
        shutil.copyfile(jpg_file, dst_bmp_path[:-4] + '.jpg')
        if cnt > 3:
            break