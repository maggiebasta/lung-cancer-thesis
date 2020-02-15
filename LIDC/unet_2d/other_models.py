from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import (
    Activation,
    add,
    BatchNormalization,
    concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    ReLU,
    UpSampling2D
)


"""
A Keras/Tensorflow implementation of the original U-Net architecture
described by Olaf Ronneberger et. al in "U-Net: Convolutional Networks for
Biomedical Image Segmentation":
paper: https://arxiv.org/abs/1505.04597
Adapted and simplified from:
https://github.com/zhixuhao/unet/blob/master/data.py
and:
https://github.com/a-martyn/unet/blob/master/model/data_loader.py
"""


def encoder_block(x, filters, kernel_size, downsample=False):
    conv_kwargs = dict(
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )

    # Downsample input to halve Height and Width dimensions
    if downsample:
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # Convolve
    x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
    x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
    return x


def decoder_block(inputs, filters, kernel_size, transpose=True):
    x, shortcut = inputs

    conv_kwargs = dict(
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )

    # Upsample input to double Height and Width dimensions
    if transpose:
        # Transposed convolution a.k.a fractionally-strided convolution
        # or deconvolution although use of the latter term is confused.
        # Excellent explanation: https://github.com/vdumoulin/conv_arithmetic
        up = Conv2DTranspose(filters, 2, strides=2, **conv_kwargs)(x)
    else:
        # Upsampling by simply repeating rows and columns then convolve
        up = UpSampling2D(size=(2, 2))(x)
        up = Conv2D(filters, 2, **conv_kwargs)(up)

    # Concatenate u-net shortcut to input
    x = concatenate([shortcut, up], axis=3)

    # Convolve
    x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
    x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
    return x


# INTENDED API
# ----------------------------------------------------------------------------

def unet(input_size=(256, 256, 1), output_channels=1, transpose=True):
    """
    U-net implementation adapted translated from authors original
    source code available here: 
    https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    """

    # The U
    inputs = Input(input_size)
    e1 = encoder_block(inputs, 64, 3, downsample=False)
    e2 = encoder_block(e1, 128, 3, downsample=True)
    e3 = encoder_block(e2, 256, 3, downsample=True)
    e4 = encoder_block(e3, 512, 3, downsample=True)
    e4 = Dropout(0.5)(e4)

    e5 = encoder_block(e4, 1024, 3, downsample=True)
    e5 = Dropout(0.5)(e5)

    d6 = decoder_block([e5, e4], 512, 3, transpose=transpose)
    d7 = decoder_block([d6, e3], 256, 3, transpose=transpose)
    d8 = decoder_block([d7, e2], 128, 3, transpose=transpose)
    d9 = decoder_block([d8, e1], 64,  3, transpose=transpose)

    # Ouput
    op = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(d9)
    op = ReLU()(op)
    op = Conv2D(output_channels, 1)(op)
    op = Activation('sigmoid')(op)

    # Build
    model = Model(inputs=[inputs], outputs=[op])
    return model


"""
2D Residual U-Net for lung nodule segmentation.
Adapted from:
https://github.com/DuFanXin/deep_residual_unet/blob/master/res_unet.py
"""


def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(
        filters=nb_filters[0],
        kernel_size=(3, 3),
        padding='same',
        strides=strides[0]
    )(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(
        filters=nb_filters[1],
        kernel_size=(3, 3),
        padding='same',
        strides=strides[1]
    )(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        strides=(1, 1)
    )(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        strides=(1, 1)
    )(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path


def original_resunet(input_shape=(256, 256, 1)):
    inputs = Input(shape=input_shape)

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder)

    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)

    return Model(inputs=[inputs], outputs=[path])