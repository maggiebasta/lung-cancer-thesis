import os
import sys

import cv2 as cv
import numpy as np
import scipy
from skimage.io import imsave, imread
from skimage import measure, morphology
from sklearn.cluster import KMeans


def get_lung_mask(img):

    # threshold for haunsfield units
    #  = cv.normalize(
    #     img,
    #     np.zeros(img.shape),
    #     -1200,
    #     600,
    #     cv.NORM_MINMAX
    # )

    # Find the average pixel value near lungs to renormalize washed out images
    middle = img[100:400, 100:400]
    mean = np.mean(img)
    mean = np.mean(middle)
    max_im = np.max(img)
    min_im = np.min(img)

    # Moving underflow and overflow on pixel spectrum to improve thresholding
    img[img == max_im] = mean
    img[img == min_im] = mean

    # Use Kmeans to separate foreground (radio-opaque tissue) and background
    # (radio transparent tissue ie lungs)
    # Do only on the center of the image to avoid non-tissue parts of the image
    kmeans = KMeans(n_clusters=2).fit(
        np.reshape(middle, [np.prod(middle.shape), 1])
    )
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # Erosion to remove graininess from some of the regions
    eroded = morphology.erosion(thresh_img, np.ones([2, 2]))

    # Dialation to make the lung region engulf the vessels and incursions into
    # the lung cavity by radio opaque tissue
    dilation = morphology.dilation(eroded, np.ones([10, 10]))

    #  1. Label each region and obtain the region properties
    #  2. Background region removed by removing regions w/ large bbox
    #  3. Remove regions close to top and bottom of image (far from lungs)
    labels = measure.label(dilation)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0] < 475 and B[3]-B[1] < 475 and B[0] > 40 and B[2] < 475:
            good_labels.append(prop.label)
    mask = np.ndarray([512, 512], dtype=np.int8)
    mask[:] = 0

    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
        mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

    mask[mask > 0] = 1

    # fill holes
    mask = np.float32(scipy.ndimage.morphology.binary_fill_holes(mask))

    for i in np.arange(2, 24, 2):
        eroded_mask = morphology.erosion(mask, np.ones([24-i, 24-i]))
        if eroded_mask.sum() != 0:
            return eroded_mask
    return mask

def normalize(img):
    return cv.normalize(
        img,
        np.zeros(img.shape),
        0,
        255,
        cv.NORM_MINMAX
    ).astype('float')


def resize(img):
    return cv.resize(img, dsize=(256, 256)).astype(np.uint8)


def preprocess(datapath, processedpath):
    os.mkdir(processedpath)
    os.mkdir(f"{processedpath}/image")
    os.mkdir(f"{processedpath}/label")

    idxs = range(len(os.listdir(f"{datapath}/image/")))
    n = len(idxs)
    for i, idx in enumerate(idxs):
        sys.stdout.write(f"\rProcessing...{i+1}/{n}")
        sys.stdout.flush()
        img = imread(f"{datapath}/image/{idx}.tif")
        mask = resize(get_lung_mask(img).astype('float'))
        img = normalize(img)
        img = resize(img)
        img = img*mask
        imsave(f"{processedpath}/image/{idx}.tif", img)

        mask = imread(f"{datapath}/label/{idx}.tif")
        mask = resize(mask)
        imsave(f"{processedpath}/label/{idx}.tif", mask)
    print(f"\nComplete.")
