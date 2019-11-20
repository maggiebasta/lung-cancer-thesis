import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img
)

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
    data_format='channels_last',
    validation_split=0.0
)
label_generator_train = ImageDataGenerator(
    rotation_range=2,
    # No brightness transform on target mask
    # No rescale transform on target mask
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode='reflect',
    data_format='channels_last',
    validation_split=0.0
)

image_generator_test = ImageDataGenerator(
    rescale=1./255,
    fill_mode='reflect',
    data_format='channels_last',
    validation_split=0.0
)

label_generator_test = ImageDataGenerator(
    # No rescale transform on target mask
    fill_mode='reflect',
    data_format='channels_last',
    validation_split=0.0
)


# Instantiated joined image and mask generators for model input
# ----------------------------------------------------------------------------

def generator(directory, input_gen, target_gen, batch_sz=2, img_sz=(256, 256)):

    input_subdir = 'image'
    label_subdir = 'label'

    # Input generator
    x_gen = input_gen.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="grayscale",
        classes=[input_subdir],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )

    # Target generator
    y_gen = input_gen.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="grayscale",
        classes=[label_subdir],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )

    generator = zip(x_gen, y_gen)
    for (x, y) in generator:
        yield (x, y)


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
