import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import (
    Input, Conv3D, UpSampling3D, BatchNormalization, Activation, add, concatenate, Lambda, multiply
)

"""
https://github.com/DuFanXin/deep_residual_unet/blob/master/res_unet.py
"""


def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv3D(filters=nb_filters[0], kernel_size=(3, 3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv3D(filters=nb_filters[1], kernel_size=(3, 3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv3D(nb_filters[1], kernel_size=(1, 1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(main_path)

    shortcut = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=(1, 1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2, 1), (1, 1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2, 1), (1, 1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    main_path = UpSampling3D(size=(2, 2, 1))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=4)
    main_path = res_block(main_path, [256, 256], [(1, 1, 1), (1, 1, 1)])

    main_path = UpSampling3D(size=(2, 2, 1))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=4)
    main_path = res_block(main_path, [128, 128], [(1, 1, 1), (1, 1, 1)])

    main_path = UpSampling3D(size=(2, 2, 1))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=4)
    main_path = res_block(main_path, [64, 64], [(1, 1, 1), (1, 1, 1)])

    return main_path


def unet3d_res(input_shape=(256, 256, 8, 1)):
    inputs = Input(shape=input_shape)

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[2], [512, 512], [(2, 2, 1), (1, 1, 1)])

    path = decoder(path, from_encoder=to_decoder)

    path = Conv3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid')(path)
    # effecitively "apply lung field mask"
    inputs2 = Lambda(lambda x: tf.math.ceil(K.clip(x, 0, 1)))(inputs)
    path = multiply([path, inputs2])

    return Model(inputs=[inputs], outputs=[path])