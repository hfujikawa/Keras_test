# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:38:46 2018

@author: hfuji
"""

# https://python-guideja.readthedocs.io/ja/latest/scenarios/xml.html
import xmltodict
import glob
import json
#from collections import OrderedDict

src_dir = 'D:/Develop/RoadDamageDetector-tf/RoadDamageDataset/Numazu/JPEGImages_output/Annotations/'

files = glob.glob(src_dir + '*.xml') 

for idx, fpath in enumerate(files):
    with open(fpath) as fd:
        doc = xmltodict.parse(fd.read(), dict_constructor=dict)
        # 辞書をjsonに変換
        jsondat = json.loads(json.dumps(doc))
        if 'object' in jsondat['annotation']:
            print(idx, jsondat['annotation']['object'])
#        if idx == 43:
#            break