import os

import cv2 as cv
import numpy as np
import pandas as pd
import pydicom
import scipy
import xml.etree.ElementTree as ET

from matplotlib.path import Path
from skimage import measure, morphology
from sklearn.cluster import KMeans


def find_ct_path(raw_path, patient_id):
    """
    Given a patient i.d., returns the path to the directory with their
    CT images

    :param raw_path: path to raw data
    :param patient_id: string of patient ID of form LIDC-IDRI-XXXX
    return: path to the folder containing CT images for a given patient
    """
    patient_dir = raw_path + patient_id
    dirs = [
        os.path.join(patient_dir, d) for d in os.listdir(patient_dir)
        if os.path.isdir(os.path.join(patient_dir, d))
    ]
    dir1 = dirs[0]
    imdir1 = [
        os.path.join(dir1, d) for d in os.listdir(dir1)
        if os.path.isdir(os.path.join(dir1, d))
        and len(os.listdir(os.path.join(dir1, d))) > 1
    ][0]
    if len(dirs) == 1:
        return imdir1

    dir2 = dirs[1]
    imdir2 = [
        os.path.join(dir2, d) for d in os.listdir(dir2)
        if os.path.isdir(os.path.join(dir2, d))
    ][0]
    im1 = os.path.join(
        imdir1,
        [im for im in os.listdir(imdir1) if im.endswith('dcm')][0]
    )
    type1 = pydicom.dcmread(im1)[('0008', '0060')].value
    if type1 == 'CT':
        return imdir1
    else:
        return imdir2


def get_series_uid(dirname):
    """
    Given the path to the directory with the CT files for a patient,
    returns the series uid for the patient

    :param dirname: absolute path to the folder containing CT images
    for a given patient
    return: series uid
    """
    rootfile = [f for f in os.listdir(dirname) if f.endswith(".xml")][0]
    root = ET.parse(os.path.join(dirname, rootfile))
    header = root.find('{http://www.nih.gov}ResponseHeader')
    series_uid = header.find('{http://www.nih.gov}SeriesInstanceUid')
    return series_uid.text


def get_uids_table(dirname):
    """
    Given the path to the directory with the CT files for a patient,
    returns a dataframe containing the UIDs for all CT images in the
    directory (index=UID)

    :param dirname: absolute path to the folder containing CT images
    for a given patient
    return: DataFrame with image UID's and their absolute paths
    """
    ImPaths = [
        os.path.join(dirname, f) for f in os.listdir(dirname)
        if f.endswith('dcm')
    ]
    UIDPaths = {}
    for image_path in ImPaths:
        ds = pydicom.dcmread(image_path)
        uid = ds[('0008', '0018')].value
        pos = ds[('0020', '0032')].value
        UIDPaths[uid] = [pos, image_path]

    df = pd.DataFrame.from_dict(
        UIDPaths, orient='index', columns=['position', 'path']
    )

    df.index.name = 'UID'
    return df


def get_rois_table(dirname):
    """
    Given the path to the directory with the CT files for a patient,
    returns a dataframe containing all ROIs for all contoured images
    where index=UID

    :param dirname: absolute path to the folder containing CT images
    for a given patien
    return: list of ROI boundaries defined in the CT image XML file
    """
    ROIs = {}
    xmls = [f for f in os.listdir(dirname) if f.endswith(".xml")]
    if len(xmls) < 1:
        print(f"\nNo XML found for {dirname}\n")
    else:
        rootfile = xmls[0]
        root = ET.parse(os.path.join(dirname, rootfile))
        for session in root.findall('{http://www.nih.gov}readingSession'):
            for readNodule in session.findall('{http://www.nih.gov}unblindedReadNodule'):
                for roi in readNodule.findall('{http://www.nih.gov}roi'):
                    region = []
                    uid = roi.find('{http://www.nih.gov}imageSOP_UID').text
                    inclusion = roi.find('{http://www.nih.gov}inclusion').text
                    edgeMaps = roi.findall('{http://www.nih.gov}edgeMap')
                    if inclusion:
                        # exlude > 3mm nodules
                        if len(edgeMaps) > 1:
                            for point in edgeMaps:
                                x = point.find('{http://www.nih.gov}xCoord').text
                                y = point.find('{http://www.nih.gov}yCoord').text
                                region.append((int(x), int(y)))
                            if uid in ROIs:
                                ROIs[uid][0].append(region)
                            else:
                                ROIs[uid] = [[region]]

        # remove contours with less than 3 radiologist markings
        to_delete = []
        for uid, roi in ROIs.items():
            if len(roi[0]) < 3:
                to_delete.append(uid)

        for td in to_delete:
            del ROIs[td]
    df = pd.DataFrame.from_dict(
        ROIs,
        orient='index',
        columns=['ROIs']
    )
    df.index.name = 'UID'
    return df


def get_patient_table(raw_path, patient_id):
    """
    Given a patient ID, returns a "summarizing" dataframe for the contoured
    images of the patient - that is for each image, the UID, the ROI and the
    path to the image

    :param raw_path: path to raw data
    :param patient_id: string of patient ID of form LIDC-IDRI-XXXX
    return: dataframe with image UIDs, paths to images w/ contours and contours
    """
    ct_path = find_ct_path(raw_path, patient_id)
    df1 = get_uids_table(ct_path)
    df2 = get_rois_table(ct_path)
    return df2.join(df1, how='inner')


def get_mask(img, rois):
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

    # iteratively add roi regions to mask
    for roi in rois:

        # from roi to a matplotlib path
        path = Path(roi)
        xmin, ymin, xmax, ymax = np.asarray(
            path.get_extents(), dtype=int
        ).ravel()

        # add points to mask included in the path
        mask = np.logical_or(mask, np.array(path.contains_points(points)))

    # reshape mask
    mask = np.array([float(m) for m in mask])
    img_mask = mask.reshape(x.shape).T

    return img_mask

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
