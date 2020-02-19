import os
import sys

import cv2 as cv
import numpy as np
import pickle as pkl
from skimage import measure
from skimage.io import imread, imsave

sys.path.append("../")
from preprocess_helpers import resize, normalize


def get_most_activated_roi(y_pred_mask):
    """
    Given a predicted nodule segmentation mask, returns a pair of coordinates
    for the lower left and upper right corners of the 50 x 50 pixel ROI
    corresponding to the most activated region

    :param y_pred_mask: predicted binary mask for nodules
    """
    blobs = y_pred_mask > .1
    blobs_labels = measure.label(blobs, background=0)
    labels = np.unique(blobs_labels)[1:]
    activations = []
    for label in labels:
        blob_activation = 0.
        idxs = np.argwhere(blobs_labels == label)
        for idx_x, idx_y in idxs:
            blob_activation += y_pred_mask[idx_x][idx_y]
        activations.append(blob_activation)
    activations = np.array(activations)
    max_region = np.argwhere(blobs_labels == np.argmax(activations) + 1).T
    x_center = int(max_region[0].mean())
    y_center = int(max_region[1].mean())
    return [(x_center-25, y_center-25), (x_center+25, y_center+25)]


def get_rois(extracted_path, processed_path, roi_2d_path, roi_3d_path, model):
    """
    Given a model and original images, predicts and saves the 2D (50 x 50) and
    3D (50 x 50 x 50) ROI for the nodule

    :param extracted_path: path to the original (unprocessed) CT image slice
    :param processed_path: path to the processed CT image slice
    :param roi_path: path to save predicted ROI output
    :param model: pretrained model for nodule segementation
    """
    os.mkdir(roi_2d_path)
    os.mkdir(roi_3d_path)
    pids = os.listdir(processed_path)
    n = len(pids)
    Areas = {}
    for i, pid in enumerate(pids):
        max_area = 0
        sys.stdout.write(f"\rGetting ROIs...{i+1}/{n}")
        sys.stdout.flush()
        for im_path in os.listdir(processed_path + '/' + str(pid)):
            x = imread(
                os.path.join(processed_path + '/' + str(pid), im_path)
            ).reshape(1, 256, 256, 1)/255

            # reshape from (256, 256, 1)
            nodule_pred = model.predict(x).reshape(256, 256)
            max_area = max(max_area, nodule_pred.sum())
            try:
                # 2d
                mins, maxs = get_most_activated_roi(nodule_pred)
                xmin, ymin = mins
                xmax, ymax = maxs
                predicted_roi = resize(normalize(imread(
                    os.path.join(extracted_path + '/' + str(pid), im_path)
                )))[xmin:xmax][:, ymin:ymax]
                if not os.path.isdir(roi_2d_path + '/' + str(pid)):
                    os.mkdir(roi_2d_path + '/' + str(pid))
                imsave(f"{roi_2d_path}/{pid}/{im_path}", predicted_roi)

                # 3d
                inpath = f'data/nlst_extracted_3d/{pid}/{im_path[:-4]}.pkl'
                with open(inpath, "rb") as input_file:
                    cube = pkl.load(input_file)
                new_dims = cube.shape[1]
                nodule_pred = cv.resize(
                    nodule_pred, dsize=(new_dims, new_dims))
                mins, maxs = get_most_activated_roi(nodule_pred)
                xmin, ymin = mins
                xmax, ymax = maxs
                mins, maxs = get_most_activated_roi(nodule_pred)
                xmin, ymin = mins
                xmax, ymax = maxs
                predicted_roi = np.array([
                    normalize(slc)[xmin:xmax][:, ymin:ymax]
                    for slc in cube
                ])
                if not os.path.isdir(roi_3d_path + '/' + str(pid)):
                    os.mkdir(roi_3d_path + '/' + str(pid))
                outpath = f"{roi_3d_path}/{pid}/{im_path[:-4]}.pkl"
                pkl.dump(predicted_roi, open(outpath, "wb"))
            except (ValueError, IndexError):
                sys.stdout.write(f"\nNo predicted ROI for {pid} {im_path}\n")
                pass
        Areas[pid] = max_area
    pkl.dump(Areas, open('data/areas.pkl', "wb"))
    print(f"\nComplete.")
