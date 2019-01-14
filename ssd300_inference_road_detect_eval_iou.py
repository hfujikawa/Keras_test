# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 14:33:40 2018
ssd_keras/ssd300_inference.ipynb
@author: hfuji
"""

import os
import shutil

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

#%matplotlib inline

MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)
#specific_iou_flagged = False

def eval_iou(class_name, bb, ground_truth_data):
  tp = 0
  fp = 0
  ovmax = -1
  gt_match = -1
  # load prediction bounding-box
#  bb = [ float(x) for x in prediction["bbox"].split() ]
  for ith, obj in enumerate(ground_truth_data):
    # look for a class_name match
#    if obj["class_name"] == class_name:
    if obj[0] == class_name:
#      bbgt = [ float(x) for x in obj["bbox"].split() ]
      bbgt = [obj[1], obj[2], obj[3], obj[4]]
      print('bbgt: ', bbgt)
      bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
      iw = bi[2] - bi[0] + 1
      ih = bi[3] - bi[1] + 1
      if iw > 0 and ih > 0:
        # compute overlap (IoU) = area of intersection / area of union
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
        ov = iw * ih / ua
        if ov > ovmax:
          ovmax = ov
          gt_match = obj
          gt_match[6] = ith
          print("Candidate Match!: ", ov, obj)

  # assign prediction as true positive/don't care/false positive
  status = "NO MATCH FOUND!" # status is only used in the animation
  # set minimum overlap
  min_overlap = MINOVERLAP
  if ovmax >= min_overlap:
#    if "difficult" not in gt_match:
#        if not bool(gt_match["used"]):
        if not bool(gt_match[5]):
          # true positive
          tp = 1
#          gt_match["used"] = True
#          gt_match[5] = True
          gt_match[5] = 1
#          count_true_positives[class_name] += 1
          # update the ".json" file
          status = "MATCH!"
        else:
          # false positive (multiple detection)
          fp = 1
          status = "REPEATED MATCH!"
  else:
    # false positive
    fp = 1
    if ovmax > 0:
      status = "INSUFFICIENT OVERLAP"
      
  return tp, fp, status
#  return tp, fp, gt_match[6], status

# Set the image size.
img_height = 300
img_width = 300

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = 'D:/Develop/weights/ssd300_raccoon_freeze_epoch-30_loss-2.8218_val_loss-3.4306.h5'
weights_path = 'D:/Develop\models/VOC0712\SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.h5'
weights_path = 'ssd300_road_detect_epoch-06_loss-6.2384_val_loss-5.8475.h5'
weights_path = 'ssd300_road_detect_epoch-44_loss-5.1428_val_loss-5.3631.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# Display the image and draw the predicted boxes onto it.

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
#classes = ['background', 'raccoon']
#classes = ['background', 'bicycle', 'bus', 'car', 'person']
classes = ['background', 'D00', 'D01', 'D10', 'D11', 'D20', 'D30', 'D40', 'D43', 'D44']

# Create a `BatchGenerator` instance and parse the Pascal VOC labels.

dataset = DataGenerator()

# TODO: Set the paths to the datasets here.

dataset_dir = 'D:/Develop/data/VOCdevkit/VOC2007/'
dataset_dir = 'D:/Develop/Samples/RoadDamageDetector-tf/RoadDamageDataset/Numazu/'
VOC_2007_images_dir = dataset_dir + 'JPEGImages/'
VOC_2007_annotations_dir = dataset_dir + 'Annotations/'
VOC_2007_test_image_set_filename = dataset_dir + 'ImageSets/Main/val.txt'
with open(VOC_2007_test_image_set_filename, 'rt') as fp:
    lines = fp.readlines()
    test_list = []
    for line in lines:
        line = line.strip()
        test_list.append(line)

dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                  image_set_filenames=[VOC_2007_test_image_set_filename],
                  annotations_dirs=[VOC_2007_annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=True,
                  ret=False)

convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)
BATCH_SIZE = 8
generator = dataset.generate(batch_size=BATCH_SIZE,
                             shuffle=True,
                             transformations=[convert_to_3_channels,
                                              resize],
                             returns={'processed_images',
                                      'filenames',
                                      'inverse_transform',
                                      'original_images',
                                      'original_labels'},
                             keep_images_without_gt=False)


out_dir = 'D:/Develop/Tools/mAP/'
out_lines = []
#for ite in range(len(test_list) // BATCH_SIZE):
for ite in range(1):
    # Generate a batch and make predictions.
    
    batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(generator)
    i = 0 # Which batch item to look at
    for i in range(BATCH_SIZE):
        print("Image:", batch_filenames[i])
        print()
        print("Ground truth boxes:\n")
        print(np.array(batch_original_labels[i]))
        
        # Predict.
        
        y_pred = model.predict(batch_images)
        
#        confidence_threshold = 0.5
        confidence_threshold = 0.1
        
        # Perform confidence thresholding.
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
        
        # Convert the predictions for the original image.
        y_pred_thresh_inv = apply_inverse_transforms(y_pred_thresh, batch_inverse_transforms)
        
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh_inv[i])
        
        # Display the image and draw the predicted boxes onto it.
        
        # Set the colors for the bounding boxes
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        
        plt.figure(figsize=(20,12))
        plt.imshow(batch_original_images[i])
        
        current_axis = plt.gca()
        
        basename = os.path.basename(batch_filenames[i])
        gt_list = []
        ground_truth_data = []
        for box in batch_original_labels[i]:
            xmin = box[1]
            ymin = box[2]
            xmax = box[3]
            ymax = box[4]
            label = '{}'.format(classes[int(box[0])])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
            line = str('{} {} {} {} {}'.format(label, xmin, ymin, xmax, ymax))
            gt_list.append(line)
            ground_truth_data.append([label, xmin, ymin, xmax, ymax, 0, -1])
        out_gt_dir = out_dir + 'ground-truth/'
        out_gt_path = out_gt_dir + basename[:-4] + '.txt'
        with open(out_gt_path, 'wt') as fp:
            fp.write('\n'.join(gt_list))
        
        predict_list = []
        tp_cnt = 0
        fp_cnt = 0
        fn_cnt = len(ground_truth_data)
        for box in y_pred_thresh_inv[i]:
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
            line_pred = str('{} {:.2f} {} {} {} {}'.format(classes[int(box[0])], box[1], int(xmin), int(ymin), int(xmax), int(ymax)))
            predict_list.append(line_pred)
            # IoU evaluation
            bbox = [box[2], box[3], box[4], box[5]]
            tp, fp, status = eval_iou(classes[int(box[0])], bbox, ground_truth_data)
            tp_cnt += tp
            fp_cnt += fp
#            tp, fp, ith, status = eval_iou(classes[int(box[0])], bbox, ground_truth_data)
            print(' eval IoU: ', tp, fp, status)
        fn_cnt -= tp_cnt
        out_line = '{},{},{},{}'.format(basename[:-4], tp_cnt, fp_cnt, fn_cnt)
        print('file, tp, fp, fn: ', out_line)
        out_lines.append(out_line)
        
        out_predict_dir = out_dir + 'predicted/'
        out_predict_path = out_predict_dir + basename[:-4] + '.txt'
        with open(out_predict_path, 'wt') as fp:
            fp.write('\n'.join(predict_list))
        
        out_img_path = out_dir + 'images/' + basename[:-4] + '.jpg'
        shutil.copyfile(batch_filenames[i], out_img_path)
        plt.figure()
        plt.show()
with open('eval.csv', 'wt') as fp:
    fp.write('\n'.join(out_lines))
