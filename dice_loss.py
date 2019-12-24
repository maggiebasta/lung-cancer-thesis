import tensorflow as tf
from keras import backend as K


"""
Dice coefficient loss function:
Implements dice coefficient loss for Keras models

Adapted from:
https://www.kaggle.com/c/ultrasound-nerve-segmentation/forums/t/21358/0-57-deep-learning-keras-tutorial
"""


def dice_coef(y_true, y_pred, smooth=10):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_weighted(y_true, y_pred, smooth=0):
    intersection = K.sum(.5*y_true * 2*y_pred, axis=[1, 2, 3])
    union = 2*K.sum(y_true, axis=[1, 2, 3]) + .5*K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_weighted_loss(y_true, y_pred):
    return 1 - dice_coef_weighted(y_true, y_pred)


def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels
    within an image because it gives all classes equal weight.
    However, it is not the defacto standard for image segmentation.
    For example, assume you are trying to predict if each pixel is cat,
    dog, or background. You have 80% background pixels, 10% dog, and
    10% cat. If the model predicts 100% background should it be be 80%
    right (as with categorical cross entropy) or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges
    on zero. This has been shifted so it converges on 0 and is smoothed
    to avoid exploding or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    # Source:
    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


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
        pos_weight=750
    )
    return tf.reduce_mean(loss)
