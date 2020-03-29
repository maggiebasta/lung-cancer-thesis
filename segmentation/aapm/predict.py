import numpy as np
from skimage import measure


def get_most_activated_roi(y_pred_mask):
    """
    Given a predicted nodule segmentation mask, returns the center
    of the most activated region

    :param y_pred_mask: predicted binary mask for nodules
    """

    y_pred_mask = y_pred_mask.reshape(256, 256)
    blobs = y_pred_mask > .1
    blobs_labels = measure.label(blobs, background=0)
    labels = np.unique(blobs_labels)[1:]
    activations = []
    for label in labels:
        blob_activation = 0.
        idxs = np.argwhere(blobs_labels == label)
        for idx_x, idx_y in idxs:
            blob_activation += y_pred_mask[idx_x][idx_y]
#         activations.append(blob_activation)
        activations.append(blob_activation/len(idxs))
    activations = np.array(activations)
    max_region = np.argwhere(blobs_labels == np.argmax(activations) + 1).T
    x_center = int(max_region[0].mean())
    y_center = int(max_region[1].mean())

    return x_center, y_center

