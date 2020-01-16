import tensorflow as tf
from tensorflow.keras.models import *

from tensorflow.keras.layers import (
    Input, Conv2D, Conv3D, BatchNormalization,
    Activation, Dense, MaxPooling2D, MaxPooling3D,
    Flatten, Dropout, LeakyReLU
)


# def cnn_baseline(input_shape=(256, 256, 1)):
#     inputs = Input(shape=input_shape)

#     path = Conv2D(filters=32, kernel_size=(3, 3))(inputs)
#     path = Activation('relu')(path)
#     path = MaxPooling2D(pool_size=(2, 2))(path)

#     path = Conv2D(filters=32, kernel_size=(3, 3))(inputs)
#     path = Activation('relu')(path)
#     path = MaxPooling2D(pool_size=(2, 2))(path)

#     path = Conv2D(filters=64, kernel_size=(3, 3))(inputs)
#     path = Activation('relu')(path)
#     path = MaxPooling2D(pool_size=(2, 2))(path)

#     path = Flatten()(path)
#     path = Dense(64)(path)
#     path = Activation('relu')(path)
#     path = Dropout(0.5)(path)
#     path = Dense(1)(path)
#     path = Activation('sigmoid')(path)

#     return Model(inputs=[inputs], outputs=[path])


def cnn_baseline(input_shape=(256, 256, 1)):
    inputs = Input(shape=input_shape)

    path = Conv2D(filters=64, kernel_size=(2, 2))(inputs)
    # path = Activation('relu')(path)
    path = LeakyReLU(alpha=.1)(path)

    path = Conv2D(filters=32, kernel_size=(3, 3))(inputs)
    # path = Activation('relu')(path)
    path = LeakyReLU(alpha=.1)(path)
    path = MaxPooling2D(pool_size=(3, 3))(path)

    path = Conv2D(filters=32, kernel_size=(3, 3))(inputs)
    # path = Activation('relu')(path)
    path = LeakyReLU(alpha=.1)(path)

    path = MaxPooling2D(pool_size=(3, 3))(path)

    path = Flatten()(path)
    path = Dense(8)(path)
    # path = Activation('relu')(path)
    path = LeakyReLU(alpha=.1)(path)
    path = Dropout(0.5)(path)
    path = Dense(1)(path)
    path = Activation('sigmoid')(path)

    return Model(inputs=[inputs], outputs=[path])


def cnn_baseline_3d(input_shape=(50, 50, 50, 1)):
    inputs = Input(shape=input_shape)

    path = Conv3D(filters=64, kernel_size=(3, 3, 3))(inputs)
    path = Activation('relu')(path)

    # path = Conv3D(filters=128, kernel_size=(3, 3, 3))(inputs)
    # path = Activation('relu')(path)
    # path = MaxPooling3D(pool_size=(3, 3, 3))(path)

    path = Conv3D(filters=256, kernel_size=(3, 3, 3))(inputs)
    path = Activation('relu')(path)

    path = Conv3D(filters=512, kernel_size=(3, 3, 3))(inputs)
    path = Activation('relu')(path)
    path = MaxPooling3D(pool_size=(3, 3, 3))(path)

    path = Conv3D(filters=32, kernel_size=(3, 3, 3))(inputs)
    path = Activation('relu')(path)

    path = MaxPooling3D(pool_size=(3, 3, 3))(path)

    path = Flatten()(path)

    path = Dense(1024)(path)
    path = Activation('relu')(path)
    # path = Dropout(0.25)(path)

    path = Dense(512)(path)
    path = Activation('relu')(path)
    # path = Dropout(0.25)(path)

    path = Dense(256)(path)
    path = Activation('relu')(path)
    path = Dropout(0.25)(path)

    path = Dense(1)(path)
    path = Activation('sigmoid')(path)

    return Model(inputs=[inputs], outputs=[path])
