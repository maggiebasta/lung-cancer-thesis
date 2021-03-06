import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import (
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
        interpolation='nearest',
        subset='training'
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
        interpolation='nearest',
        subset='training'
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
        ax.imshow(batch[0], cmap='bone')
        ax.axis('off')
        i += 1
        if i > n_rows*n_cols:
            break
    plt.show()
    return

