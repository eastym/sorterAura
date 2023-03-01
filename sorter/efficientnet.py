from tensorflow.python import keras

import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

import datetime
from tensorflow.keras import layers

num_classes = 10

feature_extractor_url = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"

feature_extractor_layer = hub.KerasLayer(feature_extractor_url,input_shape=(224,224,3))

feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])