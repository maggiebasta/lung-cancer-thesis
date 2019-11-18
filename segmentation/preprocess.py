import numpy as np
import scipy.ndimage as ndimage

from scipy import ndimage as ndi
from skimage.filters import gabor
from skimage.morphology import watershed
from skimage.feature import peak_local_max


def gaussian_smooth(img):
    """
    Given an image returns the smoothed image after applying
    a gaussian filter

    :param img: the image to smooth
    :return: the smoothed image
    """
    return(ndimage.gaussian_filter(img, sigma=1))


def gabor_filter(img):
    """
    Given an image returns the filtered image after applying
    a gabor filter

    :param img: the image to filtered
    :return: the filtered image
    """
    img_filt_real, img_filt_imag = gabor(img, frequency=0.65)
    return img_filt_real


def watershed_segment(img):
    """
    Given an image, returns the segmented image after applying
    watershed thresholding
    """
    distance = ndi.distance_transform_edt(img)
    local_maxi = peak_local_max(
        distance,
        indices=False,
        footprint=np.ones((3, 3)),
        labels=img
    )
    markers = ndi.label(local_maxi)[0]
    return watershed(-distance, markers, mask=img)

def preprocess(input_path, output_path):
    pass