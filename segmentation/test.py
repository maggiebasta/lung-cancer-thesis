import os
import sys 

import matplotlib.pyplot as plt
from skimage.io import imread
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from data_generator import *
from model import *
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img
)
print(tf.__version__)

sys.path.append("../")
from metrics import *

TRAIN_PATH = 'segmenation/data/train/'
VAL_PATH = 'segmenation/data/test/'

MODEL_NAME = 'unet_lidc.hdf5'


image_generator = ImageDataGenerator(
    rotation_range=12,
    rescale=1./255,
    shear_range=.1,
    zoom_range=.15,
    brightness_range=[.85, 1.0],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.0
)

label_generator = ImageDataGenerator(
    rotation_range=12,
    shear_range=.1,
    zoom_range=.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.0
)


train_generator = generator(
    TRAIN_PATH,
    image_generator,
    label_generator, 
    batch_sz=12
)
val_generator = generator(
    VAL_PATH,
    image_generator,
    label_generator, 
    batch_sz=8
)


model = unet_2d_res()
model.compile(
    optimizer = Adam(lr = 1e-4),
    loss=weighted_cross_entropy,
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.SpecificityAtSensitivity(.5),
    ]
)
model_checkpoint = ModelCheckpoint(MODEL_NAME, monitor='loss',verbose=0, save_best_only=True)


model_history = model.fit_generator(
    train_generator,
    validation_data= val_generator,
    validation_steps=500//8,
    steps_per_epoch=1000,
    epochs = 20,
    callbacks=[model_checkpoint]
)
# model_history = model.fit_generator(
#     train_generator,
#     validation_data= val_generator,
#     validation_steps=500//8,
#     steps_per_epoch=500,
#     epochs = 10,
#     callbacks=[model_checkpoint]
# )