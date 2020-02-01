import numpy as np
import tensorflow as tf
from skimage import measure


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
        pos_weight=12
    )
    return tf.reduce_mean(loss)


def image_recall(y_true, y_pred):
    y_pred = np.round(y_pred)
    tp = np.sum(y_true*y_pred)
    fn = np.sum(np.clip(y_true - y_pred, 0, 1))
    return np.mean(tp / (tp + fn))


def image_precision(y_true, y_pred):
    y_pred = np.round(y_pred)
    tp = np.sum(y_true*y_pred)
    fp = np.sum(np.clip(y_pred - y_true, 0, 1))
    return np.mean(tp / (tp + fp))


def dice_coef(y_true, y_pred, smooth=1):
    y_pred = np.round(y_pred)
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return np.mean((2. * intersection + smooth) / (union + smooth))


def get_most_activated_roi(y_pred_mask):
    y_pred_mask = y_pred_mask.reshape(256, 256)
    blobs = y_pred_mask > .05
    blobs_labels = measure.label(blobs, background=0)
    labels = np.unique(blobs_labels)[1:]
    activations = []
    for label in labels:
        blob_activation = 0
        idxs = np.argwhere(blobs_labels == label)
        for idx_x, idx_y in idxs:
            blob_activation += y_pred_mask[idx_x][idx_y]
        activations.append(blob_activation/len(idxs))
    max_region = np.argwhere(blobs_labels == np.argmax(activations) + 1).T
    x_center = int(max_region[0].mean())
    y_center = int(max_region[1].mean())
    return [(x_center-16, y_center-16), (x_center+16, y_center+16)]


def get_grouped_nodule_coords(y_true_mask):
    y_true_mask = y_true_mask.reshape(256, 256)
    blobs = y_true_mask == 1
    blobs_labels = measure.label(blobs, background=0)
    labels = np.unique(blobs_labels)[1:]
    return [np.argwhere(blobs_labels == l) for l in labels]


def percent_included(y_true, y_pred):
    if np.count_nonzero(np.round(y_pred)) == 0:
        return 0
    predicted_roi = get_most_activated_roi(y_pred)
    mask_groups_coords = get_grouped_nodule_coords(y_true)

    xmin, ymin = predicted_roi[0]
    xmax, ymax = predicted_roi[1]

    percentages = []
    for coords in mask_groups_coords:
        count = 0
        for xi, yi in coords:
            if xi >= xmin and xi < xmax and yi >= ymin and yi < ymax:
                count += 1
        percentages.append(count/len(coords))
    return max(percentages)
