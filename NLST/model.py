import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation, Conv3D, Dense, Dropout,
    Flatten, Input, LeakyReLU, MaxPooling3D
)


def CNN_3D(input_shape=(50, 50, 50, 1)):
    inputs = Input(shape=input_shape)

    path = Conv3D(filters=32, kernel_size=(3, 3, 3))(inputs)
    path = LeakyReLU(alpha=.1)(path)

    path = Conv3D(filters=64, kernel_size=(3, 3, 3))(path)
    path = LeakyReLU(alpha=.1)(path)
    path = MaxPooling3D(pool_size=(3, 3, 3))(path)

    path = Conv3D(filters=128, kernel_size=(3, 3, 3))(path)
    path = LeakyReLU(alpha=.1)(path)

    path = Conv3D(filters=256, kernel_size=(3, 3, 3))(path)
    path = LeakyReLU(alpha=.1)(path)
    path = MaxPooling3D(pool_size=(3, 3, 3))(path)

    path = Flatten()(path)

    path = Dense(1024)(path)
    path = LeakyReLU(alpha=.1)(path)
    path = Dropout(0.25)(path)

    path = Dense(512)(path)
    path = LeakyReLU(alpha=.1)(path)
    path = Dropout(0.25)(path)

    path = Dense(256)(path)
    path = LeakyReLU(alpha=.1)(path)
    path = Dropout(0.25)(path)

    path = Dense(1)(path)
    path = Activation('sigmoid')(path)

    return Model(inputs=[inputs], outputs=[path])


def convert_to_logits(y_pred):
    y_pred = tf.clip_by_value(
        y_pred, tf.keras.backend.epsilon(),
        1 - tf.keras.backend.epsilon()
    )

    return tf.math.log(y_pred / (1 - y_pred))


def weighted_cross_entropy(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(
        logits=y_pred,
        labels=y_true,
        pos_weight=3
    )
    return tf.reduce_mean(loss)
