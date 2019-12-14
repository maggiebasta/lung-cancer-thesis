from keras import backend as K

"""
Dice coefficient loss function:
Implements dice coefficient loss for Keras models

Adapted from:
https://www.kaggle.com/c/ultrasound-nerve-segmentation/forums/t/21358/0-57-deep-learning-keras-tutorial
"""


def dice_coef(y_true, y_pred, smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef_weighted(y_true, y_pred)


def dice_coef_weighted(y_true, y_pred, smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        10*K.sum(y_true_f) + .1*K.sum(y_pred_f) + smooth
    )


def dice_coef_weighted_loss(y_true, y_pred):
    return 1 - dice_coef_weighted(y_true, y_pred)

