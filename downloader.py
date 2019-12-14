import os
import sys

import boto3

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


def download(
    idx_i,
    idx_j,
    path='raw_data/'
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
        '0'*(4-len(n))+n for n in [str(i) for i in range(idx_i, idx_j+1)]
    ]
    ids = [f"LIDC-IDRI/LIDC-IDRI-{id_num}" for id_num in id_nums]
    for i, pid in enumerate(ids):
        sys.stdout.write(f"\rDownloading...{i+1}/{idx_j+1-idx_i}")
        sys.stdout.flush()
        keys = get_s3_keys(prefix=pid)
        if keys:
            download_from_s3(keys, path)

    print(f"\nComplete.")


if __name__ == "__main__":
    pass
