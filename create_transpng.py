# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:01:30 2018
https://www.tech-tech.xyz/archives/2715163.html
@author: hfuji
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

#orgfile = "sample/person_160.bmp"
orgfile = "D:/Develop/labelme/examples/semantic_segmentation/2011_000006.jpg"
img = cv2.imread(orgfile)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

# convert to the mask image
target = orgfile[:-4] + '*mask*.png'
files = glob.glob(target)
masks = []
for fpath in files:
    img_seg = cv2.imread(fpath)
    pure_red = np.array([0, 0, 255])
    thresh = cv2.inRange(img_seg, pure_red, pure_red)
    plt.imshow(thresh, cmap='gray')
    plt.show()
    masks.append(thresh)

# create the tranparent png image
bgr = cv2.split(img)
bgras = []
for i, alpha in enumerate(masks):
    bgra = cv2.merge(bgr + [alpha])
    plt.imshow(bgra)
    plt.show()
    cv2.imwrite("trans{}.png".format(i), bgra)
    bgras.append(bgra)

# crop the bounding rectangle
for idx, mask in enumerate(masks):
    _, contours, _hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        x,y,w,h = cv2.boundingRect(contours[i])
        print('x,y,w,h: ', x,y,w,h)
        src = bgras[idx]
        crop = src[y:y+h, x:x+w]
        
        cv2.imwrite("trans_crop{}-{}.png".format(idx, i), crop)
