import os
import shutil
import sys

import boto3
import pydicom
import lidc_helpers
from PIL import Image
from sklearn.model_selection import train_test_split


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


def _prepare(patient_id, raw_path, prepped_path, idx):
    """
    Called from download_transform_split. Given an output path, properly
    formats raw LIDC-IDRI images into prepared form and saves them in specified
    output path

    :param patient_id: patient ID of images, in form 'LIDC-IDRI-XXXX'
    :param raw_path: path to raw data to prepare
    :param prepped_path: path to directory for prepared data
    :return: None
    """

    # get image and contours for patient images, keep LARGEST countor only
    pid_df = lidc_helpers.get_patient_df(raw_path, patient_id)
    largest_pair = [None, None]
    largest_size = 0
    for row in pid_df.iterrows():
        path, rois = row[1].path, row[1].ROIs
        img = pydicom.dcmread(path).pixel_array
        mask = lidc_helpers.get_mask(img, rois)
        size = sum([sum(row) for row in mask])
        if size > largest_size:
            largest_pair = [img, mask]
            largest_size = size
    if not largest_size:
        return
    im, msk = largest_pair

    # save prepared image and mask in properly constructed directory
    while True:
        try:
            img = Image.fromarray(im)
            mask = Image.fromarray(msk)
            img.save(f"{prepped_path}/prepared/image/{idx}.tif", "TIFF")
            mask.save(f"{prepped_path}/prepared/label/{idx}.tif", "TIFF")

        except FileNotFoundError:
            if not os.path.isdir(prepped_path):
                os.mkdir(prepped_path)
            os.mkdir(f"{prepped_path}/prepared")
            os.mkdir(f"{prepped_path}/prepared/image")
            os.mkdir(f"{prepped_path}/prepared/label")
            os.mkdir(f"{prepped_path}/train")
            os.mkdir(f"{prepped_path}/train/image")
            os.mkdir(f"{prepped_path}/train/label")
            os.mkdir(f"{prepped_path}/test")
            os.mkdir(f"{prepped_path}/test/image")
            os.mkdir(f"{prepped_path}/test/label")
            continue
        break


def _build_test_train(datapath):
    """
    Called from download_transform_split. Creates training and test sets from
    prepared images

    :param datapath: the directory containing prepared, train, and test folders
    :return: None
    """
    idxs = range(len(os.listdir(f"{datapath}/prepared/image/")))
    train_idxs, test_idxs = train_test_split(idxs, test_size=.2)
    for i, idx in enumerate(train_idxs):
        im_source = f"{datapath}/prepared/image/{idx}.tif"
        im_dest = f"{datapath}/train/image/{i}.tif"
        shutil.copyfile(im_source, im_dest)

        msk_source = f"{datapath}/prepared/label/{idx}.tif"
        msk_dest = f"{datapath}/train/label/{i}.tif"
        shutil.copy(msk_source, msk_dest)

    for i, idx in enumerate(test_idxs):
        im_source = f"{datapath}/prepared/image/{idx}.tif"
        im_dest = f"{datapath}/test/image/{i}.tif"
        shutil.copyfile(im_source, im_dest)

        msk_source = f"{datapath}/prepared/label/{idx}.tif"
        msk_dest = f"{datapath}/test/label/{i}.tif"
        shutil.copy(msk_source, msk_dest)


def download_transform_split(
    num_patients,
    raw_path='data/raw/',
    prepped_path='data/',
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
        download_from_s3(keys, raw_path)

        _prepare(pid, raw_path, prepped_path, i)

        if delete_raw:
            shutil.rmtree(f"{raw_path}{pid}/")

    print(f"\nSplitting...")

    _build_test_train(prepped_path)

    if delete_raw:
        shutil.rmtree(raw_path)

    print(f"Complete.")


if __name__ == "__main__":
    num_patients = int(sys.argv[1])
    download_transform_split(num_patients)
