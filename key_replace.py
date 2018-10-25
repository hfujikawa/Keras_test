# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 22:09:50 2018
https://dev.classmethod.jp/cloud/aws/sagemaker-umaibo-object-detection/
@author: hfuji
"""
import json

fpath = "JPEGImages.json"
with open(fpath) as fp:
#    lines = fp.readlines()
#for line in lines:
#    
#    if '"0"' in line:
#        print('found')
    js = json.load(fp)
 
#    for key, val in js['frames'].items():
# 
#        k = int(key)
#        line = {}
#        line['file'] = '{0:04d}'.format(k+1) + '.jpg'
#        print(k, val)
    # https://stackoverflow.com/questions/4406501/change-the-name-of-a-key-in-dictionary
    dictionary = js['frames']
    for idx in range(10):
        old_key = str(idx)
        new_key = str(idx+100)
        print(old_key, new_key)
#        dictionary['100'] = dictionary.pop('1')
        dictionary[new_key] = dictionary.pop(old_key)
    for key, val in dictionary.items():
        print(key, val)