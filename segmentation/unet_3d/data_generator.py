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

# Instantiated joined image and mask generators for model input
# ----------------------------------------------------------------------------


def generator(directory, input_gen, target_gen, batch_sz=2, img_sz=(256, 256)):

    # Input generators
    x_generators = []
    for i in range(8):
        x_generators.append(input_gen.flow_from_directory(
            directory,
            target_size=(img_sz[0], img_sz[1]),
            color_mode="grayscale",
            classes=['image'+str(i)],
            class_mode=None,
            batch_size=batch_sz,
            seed=1,
            interpolation='nearest'
        ))

    y_generators = []
    for i in range(8):
        y_generators.append(input_gen.flow_from_directory(
            directory,
            target_size=(img_sz[0], img_sz[1]),
            color_mode="grayscale",
            classes=['label'+str(i)],
            class_mode=None,
            batch_size=batch_sz,
            seed=1,
        ))

    generators = x_generators + y_generators

    generator = zip(*generators)

    for Xy in generator:
        xs = [Xy[i] for i in range(8)]
        X = np.array(xs).reshape(batch_sz, 256, 256, 8, 1)
        ys = [Xy[i] for i in range(8, 16)]
        Y = np.array(ys).reshape(batch_sz, 256, 256, 8, 1)
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
