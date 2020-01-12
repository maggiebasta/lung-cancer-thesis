import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img
)
import sys
"""
Data Loader:
Loads the membrane cell segmentation dataset
Adapted and simplified from:
https://github.com/zhixuhao/unet/blob/master/data.py
and:
https://github.com/a-martyn/unet/blob/master/model/data_loader.py
"""

# Keras Image Data Generator Templates for train and test images and labels
# ----------------------------------------------------------------------------
image_generator_train = ImageDataGenerator(
    rotation_range=2,
    brightness_range=[0.8, 1.2],
    rescale=1./255,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode='reflect',
    # data_format='channels_last',
    validation_split=0.0
)
label_generator_train = ImageDataGenerator(
    rotation_range=2,
    # No brightness transform on target mask
    # No rescale transform on target mask
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode='reflect',
    # data_format='channels_last',
    validation_split=0.0
)

image_generator_test = ImageDataGenerator(
    rescale=1./255,
    fill_mode='reflect',
    # data_format='channels_last',
    validation_split=0.0
)

label_generator_test = ImageDataGenerator(
    # No rescale transform on target mask
    fill_mode='reflect',
    # data_format='channels_last',
    validation_split=0.0
)


# Instantiated joined image and mask generators for model input
# ----------------------------------------------------------------------------

def generator(directory, input_gen, target_gen, batch_sz=2, img_sz=(256, 256)):

    # Input generators
    x0_gen = input_gen.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="grayscale",
        classes=['image0'],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )
    x1_gen = input_gen.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="grayscale",
        classes=['image1'],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )
    x2_gen = input_gen.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="grayscale",
        classes=['image2'],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )
    x3_gen = input_gen.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="grayscale",
        classes=['image3'],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )

    # Target generators
    y0_gen = input_gen.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="grayscale",
        classes=['label0'],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )
    y1_gen = input_gen.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="grayscale",
        classes=['label1'],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )
    y2_gen = input_gen.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="grayscale",
        classes=['label2'],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )
    y3_gen = input_gen.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="grayscale",
        classes=['label3'],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )

    generator = zip(
        x0_gen, x1_gen, x2_gen, x3_gen,
        y0_gen, y1_gen, y2_gen, y3_gen
    )

    for (x0, x1, x2, x3, y0, y1, y2, y3) in generator:
        X = np.array([x0, x1, x2, x3]).reshape(batch_sz, 256, 256, 4, 1)
        Y = np.array([y0, y1, y2, y3]).reshape(batch_sz, 256, 256, 4, 1)
        # X = np.array([x0, x1, x2, x3]).reshape(batch_sz, 4, 256, 256, 1)
        # Y = np.array([y0, y1, y2, y3]).reshape(batch_sz, 4, 256, 256, 1)
        yield (X, Y)

# Data visualization
# ----------------------------------------------------------------------------

def show_augmentation(img_filepath, imageDataGenerator, n_rows=1):
    n_cols = 4
    img = load_img(img_filepath)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    fig = plt.figure(figsize=(16, 8))
    i = 1
    for batch in imageDataGenerator.flow(x, batch_size=1, seed=1):
        ax = fig.add_subplot(n_rows, n_cols, i)
        ax.imshow(batch[0])
        ax.axis('off')
        i += 1
        if i > n_rows*n_cols:
            break
    plt.show()
    return


def show_sample(generator):
    batch = next(generator)
    x = batch[0][0]
    y = batch[1][0]

    size = (5, 5)
    plt.figure(figsize=size)
    plt.imshow(x[:, :, 0], cmap='gray')
    plt.show()
    plt.figure(figsize=size)
    plt.imshow(y[:, :, 0], cmap='gray')
    plt.show()
    return
