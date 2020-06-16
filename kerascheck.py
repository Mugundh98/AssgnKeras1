# -*- coding: utf-8 -*-
"""kerascheck.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gW8SDGdi037wib8BAF_wqwX1TfVpf6Qp
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np

new_model = tf.keras.models.load_model('/content/drive/My Drive/iotassign2/my_model')

# Check its architecture
new_model.summary()

new_model.compile(loss='sparse_categorical_crossentropy',  optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

import scipy.io
all_labels = scipy.io.loadmat('/content/drive/My Drive/Iot Assign 1/imagelabels.mat')['labels'][0] - 1

test_labels = all_labels[0:200]

from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory('/content/drive/My Drive/Iot Assign 1/102flowers',
                              class_mode=None,
                              shuffle=False, target_size=(128, 128))

preds = new_model.predict_generator(test_generator)
preds_cls_idx = preds.argmax(axis=-1)

count=0
for i in range(200):
  if(preds_cls_idx[i]==test_labels[i]):
    count=count+1
acc = count/preds_cls_idx.shape[0]
print(acc)


