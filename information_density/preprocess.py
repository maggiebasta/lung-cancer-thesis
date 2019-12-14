import os
import pickle
import shutil
import sys

import pandas as pd
import pydicom
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
        images = []
        masks = []
        labels = []
        for row in pid_df.iterrows():
            path, rois = row[1].path, row[1].ROIs
            try:
                if pd.isna(rois):
                    labels.append(False)
            except ValueError:
                labels.append(True)
            im = pydicom.dcmread(path).pixel_array
            masks.append(get_mask(im, rois))
            images.append(im)

        # save prepared image and mask in properly constructed directory
        while True:
            try:
                for im, mask, label in zip(images, masks, labels):
                    idx = len(os.listdir(f"{prepped_path}/image/"))
                    imsave(f"{prepped_path}/image/{idx}.tif", im)
                    imsave(f"{prepped_path}/mask/{idx}.tif", mask)
                    pickle.dump(label, open(f"{prepped_path}/label/{idx}.pkl", 'wb'))

            except FileNotFoundError:
                os.mkdir(f"{prepped_path}")
                os.mkdir(f"{prepped_path}/image")
                os.mkdir(f"{prepped_path}/mask")
                os.mkdir(f"{prepped_path}/label")
                continue
            break


def preprocess(datapath, processedpath):
    os.mkdir(processedpath)
    os.mkdir(f"{processedpath}/image")
    os.mkdir(f"{processedpath}/mask")
    os.mkdir(f"{processedpath}/label")

    idxs = range(len(os.listdir(f"{datapath}/image/")))
    n = len(idxs)
    for i, idx in enumerate(idxs):
        sys.stdout.write(f"\rProcessing...{i+1}/{n}")
        sys.stdout.flush()
        img = imread(f"{datapath}/image/{idx}.tif")
        mask = preprocess_helpers.resize(
            preprocess_helpers.get_lung_mask(img).astype('float')
        )
        img = preprocess_helpers.normalize(img)
        img = preprocess_helpers.resize(img)
        img = img*mask
        imsave(f"{processedpath}/image/{idx}.tif", img)

        mask = imread(f"{datapath}/mask/{idx}.tif")
        mask = preprocess_helpers.resize(mask)
        imsave(f"{processedpath}/mask/{idx}.tif", mask)

        lbl_source = f"{datapath}/label/{idx}.pkl"
        lbl_dest = f"{processedpath}/label/{i}.pkl"
        shutil.copyfile(lbl_source, lbl_dest)
    print(f"\nComplete.")



def test_train_split(datapath, trainpath, testpath):
    """
    Called from download_transform_split. Creates training and test sets from
    prepared images

    :param datapath: the directory containing prepared, train, and test folders
    :return: None
    """

    os.mkdir(trainpath)
    os.mkdir(f"{trainpath}/image")
    os.mkdir(f"{trainpath}/label")
    os.mkdir(f"{trainpath}/mask")
    os.mkdir(testpath)
    os.mkdir(f"{testpath}/image")
    os.mkdir(f"{testpath}/label")
    os.mkdir(f"{testpath}/mask")

    idxs = range(len(os.listdir(f"{datapath}/image/")))
    train_idxs, test_idxs = train_test_split(idxs, test_size=.2)
    for i, idx in enumerate(train_idxs):
        im_source = f"{datapath}/image/{idx}.tif"
        im_dest = f"{trainpath}/image/{i}.tif"
        shutil.copyfile(im_source, im_dest)

        lbl_source = f"{datapath}/label/{idx}.pkl"
        lbl_dest = f"{trainpath}/label/{i}.pkl"
        shutil.copyfile(lbl_source, lbl_dest)

        msk_source = f"{datapath}/mask/{idx}.tif"
        msk_dest = f"{trainpath}/mask/{i}.tif"
        shutil.copy(msk_source, msk_dest)

    for i, idx in enumerate(test_idxs):
        im_source = f"{datapath}/image/{idx}.tif"
        im_dest = f"{testpath}/image/{i}.tif"
        shutil.copyfile(im_source, im_dest)

        lbl_source = f"{datapath}/label/{idx}.pkl"
        lbl_dest = f"{testpath}/label/{i}.pkl"
        shutil.copyfile(lbl_source, lbl_dest)

        msk_source = f"{datapath}/mask/{idx}.tif"
        msk_dest = f"{testpath}/mask/{i}.tif"
        shutil.copy(msk_source, msk_dest)



if __name__ == "__main__":
    num_patients = int(sys.argv[1])
    download_transform_split(num_patients)
