import os
import pickle
import shutil
import sys

import boto3
import cv2 as cv
import numpy as np
import pandas as pd
import pydicom

from skimage.io import imsave
from sklearn.model_selection import train_test_split

from lidc_helpers import (
    find_ct_path,
    get_mask,
    get_patient_df,
    get_series_uid
)


"""
Data Downloader:
Downloads and prepares raw LIDC-IDRI data from S3 storage
"""

def get_s3_keys(prefix, bucket='mbasta-thesis-2019'):
    """
    Gets a list of all keys in S3 bucket with prefix

    :param prefix: prefix of keys to return
    :bucket: optional, S3 bucket to get keys from
    :return: list of s3 object keys
    """

    s3 = boto3.client('s3')
    keys = []

    kwargs = {'Bucket': bucket, 'Prefix': prefix}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        if not resp.get('Contents'):
            return
        for obj in resp['Contents']:
            keys.append(obj['Key'])

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break
    return keys


def download_from_s3(keys, dest_path, bucket='mbasta-thesis-2019'):

    """
    Downloads all keys with prefix from S3

    :param prefix: prefix of keys to return
    :param dest_folder: where to store downloads, default is 'data/raw/'
    :bucket: optional, S3 bucket to get keys from
    :return: None
    """
    s3 = boto3.client('s3')
    for key in keys:
        dest_file = dest_path + key
        try:
            s3.download_file(bucket, key, dest_file)
        except FileNotFoundError:
            cur = ""
            Add = False
            for i, d in enumerate(dest_file.split('/')[:-1]):
                if i == 0:
                    cur = d
                else:
                    cur = cur + "/" + d
                if Add:
                    os.mkdir(cur)
                else:
                    if not os.path.isdir(cur):
                        os.mkdir(cur)
                        Add = True
            s3.download_file(bucket, key, dest_file)


def _prepare(patient_id, raw_path, prepped_path):
    """
    Called from download_transform_split. Given an output path, properly
    formats raw LIDC-IDRI images into prepared form and saves them in specified
    output path

    :param patient_id: patient ID of images, in form 'LIDC-IDRI-XXXX'
    :param raw_path: path to raw data to prepare
    :param prepped_path: path to directory for prepared data
    :return: None
    """

    # check if patient in LUMA
    uids = pickle.load(open( "uids.pkl", "rb" ))
    if get_series_uid(find_ct_path(raw_path, patient_id)) not in uids:
        return

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
        mask = get_mask(im, rois)
        masks.append(cv.resize(mask, dsize=(256, 256)).astype(np.uint8))
        im = cv.normalize(
            im,
            np.zeros(im.shape),
            0,
            255,
            cv.NORM_MINMAX
        )
        im = cv.resize(im, dsize=(256, 256)).astype(np.uint8)
        images.append(im)

    # save prepared image and mask in properly constructed directory
    while True:
        try:
            for im, mask, label in zip(images, masks, labels):
                idx = len(os.listdir(f"{prepped_path}/image/"))
                imsave(f"{prepped_path}/image/{idx}.png", im)
                imsave(f"{prepped_path}/mask/{idx}.png", mask)
                pickle.dump(label, open(f"{prepped_path}/label/{idx}.pkl", 'wb'))

        except FileNotFoundError:
            os.mkdir(f"{prepped_path}")
            os.mkdir(f"{prepped_path}/image")
            os.mkdir(f"{prepped_path}/mask")
            os.mkdir(f"{prepped_path}/label")
            continue
        break


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
        im_source = f"{datapath}/image/{idx}.png"
        im_dest = f"{trainpath}/image/{i}.png"
        shutil.copyfile(im_source, im_dest)

        lbl_source = f"{datapath}/label/{idx}.pkl"
        lbl_dest = f"{trainpath}/label/{i}.pkl"
        shutil.copyfile(lbl_source, lbl_dest)

        msk_source = f"{datapath}/mask/{idx}.png"
        msk_dest = f"{trainpath}/mask/{i}.png"
        shutil.copy(msk_source, msk_dest)

    for i, idx in enumerate(test_idxs):
        im_source = f"{datapath}/image/{idx}.png"
        im_dest = f"{testpath}/image/{i}.png"
        shutil.copyfile(im_source, im_dest)

        lbl_source = f"{datapath}/label/{idx}.pkl"
        lbl_dest = f"{testpath}/label/{i}.pkl"
        shutil.copyfile(lbl_source, lbl_dest)

        msk_source = f"{datapath}/mask/{idx}.png"
        msk_dest = f"{testpath}/mask/{i}.png"
        shutil.copy(msk_source, msk_dest)


def download_and_extract(
    num_patients,
    raw_path='data/raw/',
    prepped_path='data/prepared',
    delete_raw=True
):
    """
    Called from main method. Given a number of samples, downloads and prepares
    raw data from S3 for the Unet model.

    :param num_patients: total sample size to download from s3 and transform
    :param raw_path: optional, path for raw data, default='data/raw/'
    :param preppred_path: optional, path for prepared data, default='data/'
    :param delete_raw: optional, whether to delete raw data after, default=True
    """
    id_nums = [
        '0'*(4-len(n))+n for n in [str(i) for i in range(1, num_patients+1)]
    ]
    ids = [f"LIDC-IDRI/LIDC-IDRI-{id_num}" for id_num in id_nums]
    for i, pid in enumerate(ids):
        sys.stdout.write(f"\rPreparing...{i+1}/{num_patients}")
        sys.stdout.flush()
        keys = get_s3_keys(prefix=pid)
        if keys:
            download_from_s3(keys, raw_path)

            _prepare(pid, raw_path, prepped_path)

            if delete_raw:
                shutil.rmtree(f"{raw_path}{pid}/")

    if delete_raw:
        shutil.rmtree(raw_path)

    print(f"\nComplete.")


if __name__ == "__main__":
    num_patients = int(sys.argv[1])
    download_transform_split(num_patients)
