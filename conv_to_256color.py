# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:28:02 2018
https://stackoverflow.com/questions/3114925/pil-convert-rgb-image-to-a-specific-8-bit-palette
@author: hfuji
"""

from __future__ import division
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

palette = []
levels = 8
stepsize = 256 // levels
for i in range(256):
    v = i // stepsize * stepsize
    palette.extend((v, v, v))

assert len(palette) == 768

fname = 'D:/Develop/imgAnnotation_win_bin/sample/bike_246.mask.0.png'
im = Image.open(fname) # Replace with your image name here
# Get the colour palette
palette256 = im.getpalette()

original_path = 'D:/Develop/data/tmp/image2.jpg'
original_path = 'D:/Develop/imgAnnotation_win_bin/sample/ladder_001.mask.0b.bmp'
original = Image.open(original_path)
original_arr = np.array(original)
plt.imshow(original_arr)
plt.show()
'''
converted = Image.new('P', original.size)
converted.putpalette(palette256)
colors = converted.convert('RGB').getcolors() #this converts the mode to RGB
#converted.paste(original, (0, 0))
#converted.show()
converted.save('converted.png', "PNG")
'''
# https://stackoverflow.com/questions/32323922/how-to-convert-a-24-color-bmp-image-to-16-color-bmp-in-python
newimg = original.convert(mode='P', colors=256)
#newimg.putpalette(palette256)
newimg_arr = np.array(newimg)
plt.imshow(newimg_arr)
plt.show()
newimg.save('newimg256.png')
