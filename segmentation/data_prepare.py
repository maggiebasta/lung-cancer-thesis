import os

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

    kwargs = {'Bucket': bucket, 'Prefix':prefix}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            keys.append(obj['Key'])

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break
    return keys

def download_from_s3(prefix, dest_folder='data/raw/', bucket='mbasta-thesis-2019'):
    """
    Downloads all keys with prefix from S3
    :param prefix: prefix of keys to return
    :param dest_folder: optional, where to store downloaded objects
    :bucket: optional, S3 bucket to get keys from
    :return: None 
    """
    s3 = boto3.client('s3')
    keys = get_s3_keys(prefix=prefix)
    for key in keys: 
        dest_file = dest_folder + key
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


def prepare(raw_path='data/raw/LIDC-IDRI/', output_path='data/initial/'):
    """
    Given an output path, properly formats LIDC-IDRI images into training and
    test sets for the unet and saves them in the directories of the specified path

    :param raw_path: path to raw data to prepare
    :param output_path: path to directory for prepared data
    :return: None
    """
    
    data = []

    # get all patient ids 
    patient_ids = [p for p in os.listdir(raw_path)]
    
    # get image and contours for all patient images, keep LARGEST countor only
    for pid in patient_ids:
        pid_df = lidc_helpers.get_patient_df(raw_path, pid)
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
        data.append(largest_pair)

    # Split data into test and train
    train_data, test_data = train_test_split(data, test_size=.3, random_state=109)

    # Build training set
    while True:
        try: 
            count = 0
            # count = len(os.listdir(f"{output_path}/train/image"))

            # Build training set
            for im, msk in train_data:
                img = Image.fromarray(im)
                mask = Image.fromarray(msk)
                img.save(f"{output_path}/train/image/{count}.tif", "TIFF")
                mask.save(f"{output_path}/train/label/{count}.tif", "TIFF")
                count += 1

            # Build test set
            count = 0
            # count = len(os.listdir(f"{output_path}/test/image"))
            for im, msk in test_data:
                img = Image.fromarray(im)
                mask = Image.fromarray(msk)
                img.save(f"{output_path}/test/image/{count}.tif", "TIFF")
                mask.save(f"{output_path}/test/label/{count}.tif", "TIFF")
                count += 1

        # Create the output directory if it does not exist
        except FileNotFoundError:
            os.mkdir(output_path)
            os.mkdir(f"{output_path}/train")
            os.mkdir(f"{output_path}/train/image")
            os.mkdir(f"{output_path}/train/label")
            os.mkdir(f"{output_path}/test")
            os.mkdir(f"{output_path}/test/image")
            os.mkdir(f"{output_path}/test/label")
            continue
        break

if __name__ == "__main__": 
    prepare() 
  