import cv2 as cv
import numpy as np
import scipy
from matplotlib.path import Path
from skimage import measure, morphology
from sklearn.cluster import KMeans


def get_lung_mask(img):
    """
    Given a 2D axial slice from a lung CT, returns a binary mask of the lung
    regions in the image

    :param img: 512x512 raw slice
    return: 512x512 binary mask of lung regions
    """
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
        mask = morphology.dilation(mask, np.ones([10, 10]))  # final dilation

    mask[mask > 0] = 1

    # fill holes
    mask = np.float32(scipy.ndimage.morphology.binary_fill_holes(mask))

    for i in np.arange(2, 24, 2):
        eroded_mask = morphology.erosion(mask, np.ones([24-i, 24-i]))
        if eroded_mask.sum() != 0:
            return eroded_mask
    return mask


def normalize(img):
    """
    Normalizes the image to 0-255 range

    :param img: input image
    return: normalized image
    """
    return cv.normalize(
        img,
        np.zeros(img.shape),
        0,
        255,
        cv.NORM_MINMAX
    ).astype('float')


def resize(img):
    """
    Resizes the image from 512x512 to 256x256

    :param img: input image
    return: normalized image
    """
    return cv.resize(img, dsize=(256, 256)).astype(np.uint8)


def get_nodule_mask(img, rois):
    """
    Given an image and its roi (list of contour boundary points), returns a
    2D binary mask for the image

    :param img: 2D numpy array of CT image
    :param rois: 1D numpy array of list of boundary points defining ROI
    returns: 2D numpy array of image's binary contour
    """
    x, y = np.mgrid[:img.shape[1], :img.shape[0]]

    # mesh grid to a list of points
    points = np.vstack((x.ravel(), y.ravel())).T

    # empty mask
    mask = np.zeros(img.shape[0]*img.shape[1])

    try:
        # iteratively add roi regions to mask
        for roi in rois:
            # from roi to a matplotlib path
            path = Path(roi)
            xmin, ymin, xmax, ymax = np.asarray(
                path.get_extents(),
                dtype=int
            ).ravel()

            # add points to mask included in the path
            mask = np.logical_or(mask, np.array(path.contains_points(points)))

    # except if image is w/o ROIs (empty mask)
    except TypeError:
        pass

    # reshape mask
    mask = np.array([float(m) for m in mask])
    img_mask = mask.reshape(x.shape).T

    return img_mask
