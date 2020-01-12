import math
import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.utils.extmath import cartesian


"""
Dice coefficient loss function:
Implements dice coefficient loss for Keras models

Adapted from:
https://www.kaggle.com/c/ultrasound-nerve-segmentation/forums/t/21358/0-57-deep-learning-keras-tutorial
"""

AXIS = [1, 2, 3]
# AXIS = [1, 2, 3, 4]


def dice_coef(y_true, y_pred, smooth=1):
    # just in case
    y_pred = K.clip(y_pred, 0, 1)

    intersection = K.sum(y_true * y_pred, axis=AXIS)
    union = K.sum(y_true, axis=AXIS) + K.sum(y_pred, axis=AXIS)
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_weighted(y_true, y_pred, smooth=0):
    # just in case
    y_pred = K.clip(y_pred, 0, 1)

    intersection = K.sum(.25*y_true * 4*y_pred, axis=[1, 2, 3])
    union = 4*K.sum(y_true, axis=[1, 2, 3]) + .25*K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_weighted_loss(y_true, y_pred):
    return 1 - dice_coef_weighted(y_true, y_pred)


def convert_to_logits(y_pred):
    y_pred = tf.clip_by_value(
        y_pred, tf.keras.backend.epsilon(),
        1 - tf.keras.backend.epsilon()
    )

    return tf.log(y_pred / (1 - y_pred))


def weighted_cross_entropy(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(
        logits=y_pred,
        targets=y_true,
        pos_weight=10
    )
    return tf.reduce_mean(loss)


def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Source:
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * 100
