import os
import pickle
import sys

import numpy as np
import pydicom
from PIL import Image, ImageEnhance
from skimage.io import imsave, imread

from helpers import (
    find_ct_path,
    get_mask,
    get_patient_table,
    get_series_uid
)

sys.path.append('../')
sys.path.append('../../')
import helpers


def extract_train(raw_path, train_extract_path, train_ids):
    """
    Given path to raw data and an output path, extracts desired slices from
    raw LIDC-IDRI images and saves them (extracts all tumor slices)

    :param raw_path: path to raw data to prepare
    :param train_extract_path: path to directory to save extracted train data
    :param train_ids: ids for train partition from random train test split
    :return: None
    """

    # save prepared image and mask in properly constructed directory
    try:
        os.mkdir(train_extract_path)
        os.mkdir(f'{train_extract_path}/image')
        os.mkdir(f'{train_extract_path}/label')
    except FileExistsError:
        sys.stdout.write('Warning: extracted folder already exists')

    start = 1
    end = len(train_ids)
    for i, patient_id in enumerate(train_ids):
        start = 1

        sys.stdout.write(
            f'\rExtracting: {i+1}/{end+1-start}, '
            f'Total images extracted: '
            f"{len(os.listdir(f'{train_extract_path}/image/'))}"
        )
        sys.stdout.flush()
        # # check if patient in LUMA
        # uids = pickle.load(open('uids.pkl', 'rb'))
        # if not os.path.exists(raw_path + patient_id):
        #     continue
        # if get_series_uid(find_ct_path(raw_path, patient_id)) not in uids:
        #     continue

        # get image and contours for patient images
        try:
            table = get_patient_table(raw_path, patient_id)
            for row in table.iterrows():
                path, rois = row[1].path, row[1].ROIs
                img = pydicom.dcmread(path).pixel_array
                mask = get_mask(img, rois)
                idx = len(os.listdir(f'{train_extract_path}/image/'))
                imsave(f'{train_extract_path}/image/{idx}.tif', img)
                imsave(f'{train_extract_path}/label/{idx}.tif', mask)
        except:
            print(f"\nError processing {patient_id}")

    print(f'\nComplete.')


def extract_test(raw_path, test_extract_path, test_ids):
    """
    Given path to raw data, an output path, and patient ids, extracts
    desired slices for test partition from raw LIDC-IDRI images and saves
    them (extracts tumor slices w/ largest stumor area per patient only)

    :param raw_path: path to raw data to prepare
    :param test_extract_path: path to directory to save extracted test data
    :param test_ids: ids for test partition from random train test split
    :return: None
    """

    # save prepared image and mask in properly constructed directory
    try:
        os.mkdir(test_extract_path)
        os.mkdir(f'{test_extract_path}/image')
        os.mkdir(f'{test_extract_path}/label')
    except FileExistsError:
        sys.stdout.write('Warning: prepared folder already exists')

    start = 1
    end = len(test_ids)
    for i, patient_id in enumerate(test_ids):
        sys.stdout.write(
            f'\rExtracting: {i+1}/{end+1-start}, '
            f'Total images extracted: '
            f"{len(os.listdir(f'{test_extract_path}/image/'))}"
        )
        sys.stdout.flush()
        # # check if patient in LUMA
        # uids = pickle.load(open('uids.pkl', 'rb'))
        # if not os.path.exists(raw_path + patient_id):
        #     continue
        # if get_series_uid(find_ct_path(raw_path, patient_id)) not in uids:
        #     continue

        # get image and contours for patient images

        try:
            pid_df = get_patient_table(raw_path, patient_id)

            # save largest pair only
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

            idx = len(os.listdir(f'{test_extract_path}/image/'))
            imsave(f'{test_extract_path}/image/{idx}.tif', im)
            imsave(f'{test_extract_path}/label/{idx}.tif', msk)
        except:
            print(f"\nError processing {patient_id}")
    
    print(f'\nComplete.')


def preprocess_train(datapath, processedpath):
    os.mkdir(processedpath)
    os.mkdir(f'{processedpath}/image')
    os.mkdir(f'{processedpath}/label')

    idxs = range(len(os.listdir(f'{datapath}/image/')))
    n = len(idxs)
    for i, idx in enumerate(idxs):
        sys.stdout.write(f'\rProcessing...{i+1}/{n}')
        sys.stdout.flush()
        img = imread(f'{datapath}/image/{idx}.tif')

        lung_mask = helpers.get_lung_mask(img).astype('float')
        # if str(idx) + '.tif' in os.listdir('data/special_train_masks'):
        #     lung_mask += imread(
        #         f'data/special_train_masks/{idx}.tif'
        #     ).astype('float')
        #     lung_mask = np.clip(lung_mask, 0, 1)
        lung_mask = helpers.resize(lung_mask)
        if lung_mask.sum() == 0:
            sys.stdout.write(
                f'\rEmpty lung field returned for image {idx}. Skipping\n'
            )
            continue
        img = helpers.normalize(img)
        img = helpers.resize(img)
        img = img*lung_mask
        pil_im = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_im)
        enhanced_im = enhancer.enhance(2.0)
        np_im = np.array(enhanced_im)
        imsave(f'{processedpath}/image/{idx}.tif', np_im)

        mask = imread(f'{datapath}/label/{idx}.tif')
        mask = helpers.resize(mask)
        imsave(f'{processedpath}/label/{idx}.tif', mask)
    print(f'\nComplete.')


def preprocess_test(datapath, processedpath):
    os.mkdir(processedpath)
    os.mkdir(f'{processedpath}/image')
    os.mkdir(f'{processedpath}/label')

    idxs = range(len(os.listdir(f'{datapath}/image/')))
    n = len(idxs)
    for i, idx in enumerate(idxs):
        sys.stdout.write(f'\rProcessing...{i+1}/{n}')
        sys.stdout.flush()
        img = imread(f'{datapath}/image/{idx}.tif')

        lung_mask = helpers.get_lung_mask(img).astype('float')
        # if str(idx) + '.tif' in os.listdir('data/special_test_masks'):
        #     lung_mask += imread(
        #         f'data/special_test_masks/{idx}.tif'
        #     ).astype('float')
        #     lung_mask = np.clip(lung_mask, 0, 1)
        lung_mask = helpers.resize(lung_mask)

        if lung_mask.sum() == 0:
            sys.stdout.write(
                f'\rEmpty lung field returned for image {idx}. Skipping\n'
            )
            continue
        img = helpers.normalize(img)
        img = helpers.resize(img)

        img = img*lung_mask
        pil_im = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_im)
        enhanced_im = enhancer.enhance(2.0)
        np_im = np.array(enhanced_im)
        imsave(f'{processedpath}/image/{idx}.tif', np_im)

        mask = imread(f'{datapath}/label/{idx}.tif')
        mask = helpers.resize(mask)
        imsave(f'{processedpath}/label/{idx}.tif', mask)
    print(f'\nComplete.')
