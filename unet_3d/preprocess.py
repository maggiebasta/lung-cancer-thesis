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
    get_patient_df_v2,
    get_series_uid
)

sys.path.append("../")
import preprocess_helpers


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

        if not os.path.exists(raw_path + patient_id):
            continue

        # check if patient in LUMA
        uids = pickle.load(open("uids.pkl", "rb"))
        if get_series_uid(find_ct_path(raw_path, patient_id)) not in uids:
            continue

        # get image and contours for patient images
        pid_df = get_patient_df_v2(raw_path, patient_id)
        if isinstance(pid_df, type(None)):
            continue

        image = [
            pydicom.dcmread(r[1].path).pixel_array for r in pid_df.iterrows()
        ]
        rois = [row[1].ROIs for row in pid_df.iterrows()]
        mask = [get_mask(im, roi) for im, roi in zip(image, rois)]

        # save prepared image and mask in properly constructed directory
        while True:
            try:
                idx = len(os.listdir(f"{prepped_path}/image1/"))
                for i in range(4):
                    imsave(f"{prepped_path}/image{i}/{idx}.tif", image[i])
                    imsave(f"{prepped_path}/label{i}/{idx}.tif", mask[i])

            except FileNotFoundError:
                if not os.path.isdir(prepped_path):
                    os.mkdir(prepped_path)
                for i in range(4):
                    os.mkdir(f"{prepped_path}/image{i}")
                    os.mkdir(f"{prepped_path}/label{i}")
                continue
            break
    print(f"\nComplete.")


def preprocess(datapath, processedpath):
    os.mkdir(processedpath)
    os.mkdir(f"{processedpath}/image")
    os.mkdir(f"{processedpath}/label")
    for i in range(4):
        os.mkdir(f"{processedpath}/image{i}")
        os.mkdir(f"{processedpath}/label{i}")

    idxs = range(len(os.listdir(f"{datapath}/image0/")))
    n = len(idxs)
    for i, idx in enumerate(idxs):
        sys.stdout.write(f"\rProcessing...{i+1}/{n}")
        sys.stdout.flush()
        empty_found = False
        for j in range(4):
            if empty_found:
                continue
            img = imread(f"{datapath}/image{j}/{idx}.tif")
            mask = preprocess_helpers.resize(
                preprocess_helpers.get_lung_mask(img).astype('float')
            )
            if mask.sum() == 0:
                sys.stdout.write(
                    f"\rEmpty lung field returned for image {idx}. Skipping\n"
                )
                empty_found = True
                # delete previous scans for pid
                for k in range(j):
                    os.remove(f"{processedpath}/image{k}/{idx}.tif")
                continue
            img = preprocess_helpers.normalize(img)
            img = preprocess_helpers.resize(img)
            img = img*mask
            pil_im = Image.fromarray(img)
            enhancer = ImageEnhance.Contrast(pil_im)
            enhanced_im = enhancer.enhance(2.0)
            np_im = np.array(enhanced_im)
            imsave(f"{processedpath}/image{j}/{idx}.tif", np_im)

            mask = imread(f"{datapath}/label{j}/{idx}.tif")
            mask = preprocess_helpers.resize(mask)
            imsave(f"{processedpath}/label{j}/{idx}.tif", mask)
    print(f"\nComplete.")


def test_train_split(datapath, trainpath, testpath):
    """
    Called from download_transform_split. Creates training and test sets from
    prepared images

    :param datapath: the directory containing prepared, train, and test folders
    :return: None
    """

    os.mkdir(trainpath)
    os.mkdir(testpath)
    for i in range(4):
        os.mkdir(f"{trainpath}/image{i}")
        os.mkdir(f"{trainpath}/label{i}")
        os.mkdir(f"{testpath}/image{i}")
        os.mkdir(f"{testpath}/label{i}")

    idxs = os.listdir(f"{datapath}/image3/")
    train_idxs, test_idxs = train_test_split(idxs, test_size=.2)
    for i, idx in enumerate(train_idxs):
        for j in range(4):
            im_source = f"{datapath}/image{j}/{idx}"
            im_dest = f"{trainpath}/image{j}/{i}.tif"
            shutil.copyfile(im_source, im_dest)

            msk_source = f"{datapath}/label{j}/{idx}"
            msk_dest = f"{trainpath}/label{j}/{i}.tif"
            shutil.copy(msk_source, msk_dest)

    for i, idx in enumerate(test_idxs):
        for j in range(4):
            im_source = f"{datapath}/image{j}/{idx}"
            im_dest = f"{testpath}/image{j}/{i}.tif"
            shutil.copyfile(im_source, im_dest)

            msk_source = f"{datapath}/label{j}/{idx}"
            msk_dest = f"{testpath}/label{j}/{i}.tif"
            shutil.copy(msk_source, msk_dest)


if __name__ == "__main__":
    pass
