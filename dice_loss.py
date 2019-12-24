import tensorflow as tf
from keras import backend as K


"""
Dice coefficient loss function:
Implements dice coefficient loss for Keras models

Adapted from:
https://www.kaggle.com/c/ultrasound-nerve-segmentation/forums/t/21358/0-57-deep-learning-keras-tutorial
"""


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[3, 2, 1]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef_weighted(y_true, y_pred)


def dice_coef_weighted(y_true, y_pred, smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        16*K.sum(y_true_f) + .0625*K.sum(y_pred_f) + smooth
    )


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
        pos_weight=1000
    )
    return tf.reduce_mean(loss)
