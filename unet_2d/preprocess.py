import os
import pickle
import shutil
import sys

import numpy as np
import pydicom
from PIL import Image, ImageEnhance
from skimage.io import imsave, imread
from sklearn.model_selection import train_test_split

from lidc_helpers import (
    find_ct_path,
    get_mask,
    get_patient_df,
    get_series_uid
)

sys.path.append("../")
import preprocess_helpers


# def extract(raw_path, prepped_path, extract_all=True, start=None, end=None):
#     """
#     Given path to raw data and an output path, extracts desired slices from
#     sraw LIDC-IDRI images and saves them

#     :param raw_path: path to raw data to prepare
#     :param prepped_path: path to directory to save prepared data
#     :param extract_all: whether or not to extract from all raw data image sets
#     :param start: if extract_all is false, which image set to start extraction
#     :param end: if extract_all is false, which image set to end extraction
#     :return: None
#     """

#     # save prepared image and mask in properly constructed directory
#     try:
#         os.mkdir(prepped_path)
#         os.mkdir(f"{prepped_path}/image")
#         os.mkdir(f"{prepped_path}/label")
#     except FileExistsError:
#         sys.stdout.write("Warning: prepared folder already exists")

#     if extract_all:
#         start = 1
#         end = len(os.listdir(f"{raw_path}/LIDC-IDRI/"))
#     elif start is None or end is None:
#         raise ValueError("must specify start and end if extract_all is false")

#     id_nums = [
#         '0'*(4-len(n))+n for n in [str(i) for i in range(start, end+1)]
#     ]
#     ids = [f"LIDC-IDRI/LIDC-IDRI-{id_num}" for id_num in id_nums]

#     for i, patient_id in enumerate(ids):
#         sys.stdout.write(
#             f"\rExtracting: {i+1}/{end+1-start}, "
#             f"Total images extracted: "
#             f"{len(os.listdir(f'{prepped_path}/image/'))}"
#         )
#         sys.stdout.flush()
#         # check if patient in LUMA
#         uids = pickle.load(open("uids.pkl", "rb"))
#         if not os.path.exists(raw_path + patient_id):
#             continue
#         if get_series_uid(find_ct_path(raw_path, patient_id)) not in uids:
#             continue

#         # get image and contours for patient images
#         pid_df = get_patient_df(raw_path, patient_id)
#         for row in pid_df.iterrows():
#             path, rois = row[1].path, row[1].ROIs
#             img = pydicom.dcmread(path).pixel_array
#             mask = get_mask(img, rois)
#             idx = len(os.listdir(f"{prepped_path}/image/"))
#             imsave(f"{prepped_path}/image/{idx}.tif", img)
#             imsave(f"{prepped_path}/label/{idx}.tif", mask)

#     print(f"\nComplete.")

def extract(raw_path, prepped_path, extract_all=True, start=None, end=None):
    """
    Given path to raw data and an output path, extracts desired slices from
    sraw LIDC-IDRI images and saves them

    :param raw_path: path to raw data to prepare
    :param prepped_path: path to directory to save prepared data
    :param extract_all: whether or not to extract from all raw data image sets
    :param start: if extract_all is false, which image set to start extraction
    :param end: if extract_all is false, which image set to end extraction
    :return: None
    """
    if extract_all:
        start = 1
        end = len(os.listdir(f"{raw_path}/LIDC-IDRI/"))
    elif start is None or end is None:
        raise ValueError("must specify start and end if extract_all is false")

    id_nums = [
        '0'*(4-len(n))+n for n in [str(i) for i in range(start, end+1)]
    ]
    ids = [f"LIDC-IDRI/LIDC-IDRI-{id_num}" for id_num in id_nums]

    for i, patient_id in enumerate(ids):
        sys.stdout.write(f"\rExtracting...{i+1}/{end+1-start}")
        sys.stdout.flush()
        # check if patient in LUMA
        uids = pickle.load(open("uids.pkl", "rb"))
        if not os.path.exists(raw_path + patient_id):
            continue
        if get_series_uid(find_ct_path(raw_path, patient_id)) not in uids:
            continue

        # get image and contours for patient images, keep LARGEST countor only
        pid_df = get_patient_df(raw_path, patient_id)
        largest_pair = [None, None]
        largest_size = 0
        for row in pid_df.iterrows():
            path, rois = row[1].path, row[1].ROIs
            img = pydicom.dcmread(path).pixel_array
            mask = get_mask(img, rois)
            size = sum([sum(row) for row in mask])
            if size > largest_size:
                largest_pair = [img, mask]
                largest_size = size
        if not largest_size:
            continue
        im, msk = largest_pair

        # save prepared image and mask in properly constructed directory
        while True:
            try:
                idx = len(os.listdir(f"{prepped_path}/image/"))
                imsave(f"{prepped_path}/image/{idx}.tif", im)
                imsave(f"{prepped_path}/label/{idx}.tif", msk)

            except FileNotFoundError:
                if not os.path.isdir(prepped_path):
                    os.mkdir(prepped_path)
                os.mkdir(f"{prepped_path}/image")
                os.mkdir(f"{prepped_path}/label")
                continue
            break
    print(f"\nComplete.")


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
        lung_mask = preprocess_helpers.resize(
            preprocess_helpers.get_lung_mask(img).astype('float')
        )
        if lung_mask.sum() == 0:
            sys.stdout.write(
                f"\rEmpty lung field returned for image {idx}. Skipping\n"
            )
            continue
        img = preprocess_helpers.normalize(img)
        img = preprocess_helpers.resize(img)
        img = img*lung_mask
        pil_im = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_im)
        enhanced_im = enhancer.enhance(2.0)
        np_im = np.array(enhanced_im)
        imsave(f"{processedpath}/image/{idx}.tif", np_im)

        mask = imread(f"{datapath}/label/{idx}.tif")
        mask = preprocess_helpers.resize(mask)
        imsave(f"{processedpath}/label/{idx}.tif", mask)
    print(f"\nComplete.")


def test_train_split(datapath, trainpath, testpath):
    """
    Creates training and test sets from prepared images

    :param datapath: the directory containing prepared, train, and test folders
    :return: None
    """

    os.mkdir(trainpath)
    os.mkdir(f"{trainpath}/image")
    os.mkdir(f"{trainpath}/label")
    os.mkdir(testpath)
    os.mkdir(f"{testpath}/image")
    os.mkdir(f"{testpath}/label")

    idxs = os.listdir(f"{datapath}/image/")
    train_idxs, test_idxs = train_test_split(
        idxs,
        test_size=.2,
        random_state=26
    )
    for i, idx in enumerate(train_idxs):
        im_source = f"{datapath}/image/{idx}"
        im_dest = f"{trainpath}/image/{i}.tif"
        shutil.copyfile(im_source, im_dest)

        msk_source = f"{datapath}/label/{idx}"
        msk_dest = f"{trainpath}/label/{i}.tif"
        shutil.copy(msk_source, msk_dest)

    for i, idx in enumerate(test_idxs):
        im_source = f"{datapath}/image/{idx}"
        im_dest = f"{testpath}/image/{i}.tif"
        shutil.copyfile(im_source, im_dest)

        msk_source = f"{datapath}/label/{idx}"
        msk_dest = f"{testpath}/label/{i}.tif"
        shutil.copy(msk_source, msk_dest)


if __name__ == "__main__":
    pass
