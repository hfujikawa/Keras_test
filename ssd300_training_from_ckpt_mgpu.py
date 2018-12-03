# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:44:28 2018
https://keras.io/ja/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus
@author: hfuji
"""

from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from data_generator.object_detection_2d_data_generator import DataGenerator


# returns a compiled model
# identical to the previous one
model = load_model('model.h5')

freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']#,
#           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

for L in model.layers:
    if L.name in freeze:
        L.trainable = False

sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)


'''
# Replicates `model` on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=256)
'''

train_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path='dataset_raccoon_trainval.h5')
val_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path='dataset_raccoon_test.h5')

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))
