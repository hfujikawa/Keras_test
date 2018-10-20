# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 08:27:01 2018
https://utgwkk.hateblo.jp/entry/2014/04/05/235859
https://github.com/JohannesBuchner/imagehash
@author: hfuji
"""

import os
from PIL import Image
import hashlib
import imagehash
from glob import glob

flist = []
fmd5 = []
dl = []

src_path = 'D:/Develop/data/VOCdevkit/VOC2007/JPEGImages/000030.jpg'
src_hash = imagehash.average_hash(Image.open(src_path))

dirnames = ['D:/Temp/', 'D:/Develop/data/VOCdevkit/VOC2007/JPEGImages/']
dirname = 'D:/Temp/'

for dirname in dirnames:
    for e in ['bmp', 'jpg']: flist.extend(glob('%s/*.%s'%(dirname,e)))

for fn in flist:
#  with open(fn, 'rb') as fin:
#    data = fin.read()
#    m = hashlib.md5(data)
#    fmd5.append(m.hexdigest())
    ahash = imagehash.average_hash(Image.open(fn))
    fmd5.append(ahash)
    if ahash == src_hash:
        print(fn)

# フォルダ内交互比較
#for i in range(len(flist)):
#  if flist[i] in dl: continue
#  for j in range(i+1, len(flist)):
#    if flist[j] in dl: continue
#    if fmd5[i] == fmd5[j] and not flist[j] in dl:
#      print(flist[i], flist[j])
#      dl.append(flist[j])

#for a in dl: os.remove(a)