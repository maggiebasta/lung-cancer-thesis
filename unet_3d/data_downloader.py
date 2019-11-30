import os
import pickle
import shutil
import sys

import boto3
import cv2 as cv
import pydicom
import numpy as np

from scipy import ndimage, misc
from skimage.io import imsave
from sklearn.model_selection import train_test_split
from skimage.transform import rescale

from lidc_helpers import (
    find_ct_path,
    get_mask,
    get_patient_df_v2,
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


def im_norm(im):
    # normalize image
    im = cv.normalize(
        im,
        np.zeros(im.shape),
        0,
        255,
        cv.NORM_MINMAX
    )

    # resize to 256, 256
    im = cv.resize(im, dsize=(256, 256)).astype(np.uint8)

    return im


def mask_norm(mask):
    # resize to 256, 256
    return cv.resize(mask, dsize=(256, 256)).astype(np.uint8)


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

    # get image and contours for patient images
    pid_df = get_patient_df_v2(raw_path, patient_id)
    if type(pid_df) == type(None):
        return 

    image = [pydicom.dcmread(row[1].path).pixel_array for row in pid_df.iterrows()]
    rois = [row[1].ROIs for row in pid_df.iterrows()]
    mask = [get_mask(im, roi) for im, roi in zip(image, rois)]

    image = np.array([im_norm(im) for im in image])
    mask = np.array([im_norm(msk) for msk in mask])

    # save prepared image and mask in properly constructed directory
    while True:
        try:
            idx = len(os.listdir(f"{prepped_path}/image1/"))
            for i in range(4):
                imsave(f"{prepped_path}/image{i}/{idx}.png", image[i])
                imsave(f"{prepped_path}/label{i}/{idx}.png", mask[i])

        except FileNotFoundError:
            if not os.path.isdir(prepped_path):
                os.mkdir(prepped_path)
            for i in range(4):
                os.mkdir(f"{prepped_path}/image{i}")
                os.mkdir(f"{prepped_path}/label{i}")
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
    os.mkdir(testpath)
    for i in range(4):
        os.mkdir(f"{trainpath}/image{i}")
        os.mkdir(f"{trainpath}/label{i}")
        os.mkdir(f"{testpath}/image{i}")
        os.mkdir(f"{testpath}/label{i}")

    idxs = range(len(os.listdir(f"{datapath}/image0/")))
    train_idxs, test_idxs = train_test_split(idxs, test_size=.2)
    for i, idx in enumerate(train_idxs):
        for j in range(4):
            im_source = f"{datapath}/image{j}/{idx}.png"
            im_dest = f"{trainpath}/image{j}/{i}.png"
            shutil.copyfile(im_source, im_dest)

            msk_source = f"{datapath}/label{j}/{idx}.png"
            msk_dest = f"{trainpath}/label{j}/{i}.png"
            shutil.copy(msk_source, msk_dest)

    for i, idx in enumerate(test_idxs):
        for j in range(4):
            im_source = f"{datapath}/image{j}/{idx}.png"
            im_dest = f"{testpath}/image{j}/{i}.png"
            shutil.copyfile(im_source, im_dest)

            msk_source = f"{datapath}/label{j}/{idx}.png"
            msk_dest = f"{testpath}/label{j}/{i}.png"
            shutil.copy(msk_source, msk_dest)


def download_and_extract(
    idx_i,
    idx_j,
    raw_path='data/raw/',
    prepped_path='data/prepared',
    delete_raw=True
):
    """
    Called from main method. Given a number of samples, downloads and prepares
    raw data from S3 for the Unet model.

    :param idx_i: index of first patient in range to download from s3
    :param idx_j: index of last patient in range to download from s3
    :param raw_path: optional, path for raw data, default='data/raw/'
    :param preppred_path: optional, path for prepared data, default='data/'
    :param delete_raw: optional, whether to delete raw data after, default=True
    """
    id_nums = [
        '0'*(4-len(n))+n for n in [str(i) for i in range(idx_i, idx_j+1)]
    ]
    ids = [f"LIDC-IDRI/LIDC-IDRI-{id_num}" for id_num in id_nums]
    for i, pid in enumerate(ids):
        sys.stdout.write(f"\rPreparing...{i+1}/{idx_j+1-idx_i}")
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
