# -*- coding: utf-8 -*-
"""kearsAss1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WcRiWUX8FflT2QIWu0Q_bMeN6mYK87nH
"""


import tensorflow_datasets as tfds
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np

test, train, validate = tfds.load('oxford_flowers102', split=['train','test','validation'], as_supervised=True)

from tensorflow.python.framework import ops
ops.reset_default_graph()

ntrain=sum(1 for _ in train)
ntest=sum(1 for _ in test)
nval=sum(1 for _ in  validate)


def normalize_img(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (128,128))
    return image, label

mapped_train = train.map(normalize_img)
cached_train = mapped_train.cache()
batch_train = cached_train.repeat(20).batch(32)
tuned_train = batch_train.prefetch(tf.data.experimental.AUTOTUNE)

mapped_test = test.map(normalize_img)
repeated_test = mapped_test.repeat(20).batch(32)
cached_test = repeated_test.cache()
tuned_test = cached_test.prefetch(tf.data.experimental.AUTOTUNE)

mapped_validate = validate.map(normalize_img)
repeated_validate = mapped_validate.repeat(20).batch(32)
cached_validate = repeated_validate.cache()
tuned_validate = cached_validate.prefetch(tf.data.experimental.AUTOTUNE)

from keras.layers import AveragePooling2D

tf.keras.backend.clear_session()

numpy_train=tfds.as_numpy(tuned_train)
numpy_test=tfds.as_numpy(tuned_test)
numpy_validate=tfds.as_numpy(tuned_validate)
model=Sequential()
model.add(Conv2D(64, (3,3),input_shape=(128,128,3),strides=(1,1) , activation='relu'))
model.add(Conv2D(64, (3,3), padding='same',strides=(1,1), activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3,3), padding='same',strides=(1,1), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3,3), padding='same',strides=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, (3,3), padding='same',strides=(1,1), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3,3), padding='same',strides=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dense(102, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy',  optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

model.fit(numpy_train,epochs=20,steps_per_epoch=ntrain//32, validation_data=numpy_validate, validation_steps=nval//32)

scores = model.evaluate_generator(numpy_test, steps=30)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

