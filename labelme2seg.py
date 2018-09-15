# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:11:41 2018
labelme/examples/sematic_segmentation/labelme2voc.py
@author: hfuji
"""

import glob
import json
import os

import numpy as np
from PIL import Image
import cv2

def polygons_to_mask(idx, img_shape, polygons):
    global fname
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cnt_lst = []
    for xy in polygons:
        cnt_lst.append((int(xy[0]), int(xy[1])))
    cnt_arr = np.array(cnt_lst)
    
    cv2.fillPoly(mask, [cnt_arr], 255)
    height, width = img_shape[:2]
    mask_color = np.zeros([height, width, 3], dtype=np.uint8)
    mask_color[:,:,2] = mask
    cv2.imwrite("{}.mask.{}.png".format(fname,idx), mask_color)
    
    mask = np.array(mask, dtype=bool)
    return mask

def shapes_to_label(img_shape, shapes, label_name_to_value, type='class'):
    assert type in ['class', 'instance']

    cls = np.zeros(img_shape[:2], dtype=np.int32)
    if type == 'instance':
        ins = np.zeros(img_shape[:2], dtype=np.int32)
        instance_names = ['_background_']
    for idx, shape in enumerate(shapes):
        polygons = shape['points']
        label = shape['label']
        if type == 'class':
            cls_name = label
        elif type == 'instance':
            cls_name = label.split('-')[0]
            if label not in instance_names:
                instance_names.append(label)
            ins_id = len(instance_names) - 1
        cls_id = label_name_to_value[cls_name]
        mask = polygons_to_mask(idx, img_shape[:2], polygons)
        cls[mask] = cls_id
        if type == 'instance':
            ins[mask] = ins_id

    if type == 'instance':
        return cls, ins
    return cls

labels_file = 'labels.txt'
class_names = []
class_name_to_id = {}
for i, line in enumerate(open(labels_file).readlines()):
    class_id = i - 1  # starts with -1
    class_name = line.strip()
    class_name_to_id[class_name] = class_id
    if class_id == -1:
        assert class_name == '__ignore__'
        continue
    elif class_id == 0:
        assert class_name == '_background_'
    class_names.append(class_name)
class_names = tuple(class_names)
print('class_names:', class_names)

fname = 'D:\\Develop\\data\\VOCdevkit\\VOC2007\\SegmentationClass\\000063.png'
im = Image.open(fname) # Replace with your image name here
indexed = np.array(im) # Convert to NumPy array to easier access
palette = im.getpalette()

in_dir = 'D:/Develop/labelme/examples/semantic_segmentation/data_annotated'
out_dir = 'D:/Develop/labelme/examples/semantic_segmentation/data_dataset_voc'
loop = 0
for label_file in glob.glob(os.path.join(in_dir, '*.json')):
    print('Generating dataset from:', label_file)
    basename = os.path.basename(label_file)
    fname = basename[:-5]
    loop += 1
    with open(label_file) as f:
        base = os.path.splitext(os.path.basename(label_file))[0]
        out_obj_file = os.path.join(
            out_dir, 'SegmentationObjectPNG', base + '.png')

        data = json.load(f)

        img_file = os.path.join(os.path.dirname(label_file), data['imagePath'])
        img = np.asarray(Image.open(img_file))
#        Image.fromarray(img).save(out_img_file)

        lbl = shapes_to_label(
            img_shape=img.shape,
            shapes=data['shapes'],
            label_name_to_value=class_name_to_id,
        )
        print(lbl)

        img_lbl = lbl.astype(np.uint8)

        im = Image.fromarray((img_lbl).astype('uint8'), 'P')
        im.putpalette(palette)
        im.save("lbl_idx{}.png".format(loop))
