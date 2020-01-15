import os
import shutil
import sys

import numpy as np
import pickle as pkl
import pydicom
import scipy.ndimage
from PIL import Image, ImageEnhance
from skimage.io import imsave, imread
from sklearn.model_selection import train_test_split

from lidc_helpers import (
    find_ct_path,
    get_mask,
    get_patient_df_v2,
    get_series_uid
)

sys.path.append("../../")
import preprocess_helpers


def resample(image, pixel_spacing, thickness, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([thickness, pixel_spacing[0], pixel_spacing[1]])

    resize_factor = spacing / np.array(new_spacing)
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image


def extract(raw_path, prepped_path):
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
    os.mkdir(prepped_path)

    start = 1
    end = len(os.listdir(f"{raw_path}/LIDC-IDRI/"))

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
        uids = pkl.load(open("uids.pkl", "rb"))
        if get_series_uid(find_ct_path(raw_path, patient_id)) not in uids:
            continue

        # get image and contours for patient images
        pid_df = get_patient_df_v2(raw_path, patient_id)
        if isinstance(pid_df, type(None)):
            continue

        image = np.array([
            pydicom.dcmread(r[1].path).pixel_array for r in pid_df.iterrows()
        ])
        rois = [row[1].ROIs for row in pid_df.iterrows()]
        mask = np.array([get_mask(im, roi) for im, roi in zip(image, rois)])

        thickness = pydicom.dcmread(pid_df.iloc[0].path).SliceThickness
        spacing = pydicom.dcmread(pid_df.iloc[0].path).PixelSpacing[0]

        idx = len(os.listdir(f"{prepped_path}/"))
        pkl.dump(
            (image, mask, spacing, thickness),
            open(f"{prepped_path}/{idx}.pkl", 'wb')
        )

    print(f"\nComplete.")


def preprocess(datapath, processedpath):
    os.mkdir(processedpath)
    for i in range(8):
        os.mkdir(f"{processedpath}/image{i}")
        os.mkdir(f"{processedpath}/label{i}")

    idxs = os.listdir(f"{datapath}")
    n = len(idxs)
    for i, idx in enumerate(idxs):
        sys.stdout.write(f"\rProcessing...{i+1}/{n}")
        sys.stdout.flush()
        empty_found = False
        image, mask, spacing, thickness = pkl.load(
            open(f'data/extracted/{idx}', 'rb')
        )

        processed_lungmasks = []
        processed_image = []
        processed_mask = []

        for j in range(len(image)):
            if empty_found:
                continue
            img = image[j]
            processed_lungmasks.append(preprocess_helpers.get_lung_mask(img))
            processed_image.append(img)
            processed_mask.append(mask[j]*100)

        lung_mask = max(processed_lungmasks, key=lambda x: x.sum())
        image = resample(np.array(processed_image), (spacing, spacing), thickness)
        mask = resample(np.array(processed_mask), (spacing, spacing), thickness)
        mask = np.clip(mask, 0, 1)

        for k in range(8):
            im = preprocess_helpers.normalize(image[k])
            im = preprocess_helpers.resize(im)
            im = im*preprocess_helpers.resize(lung_mask)
            pil_im = Image.fromarray(im)
            enhancer = ImageEnhance.Contrast(pil_im)
            enhanced_im = enhancer.enhance(2.0)
            np_im = np.array(enhanced_im)

            mk = preprocess_helpers.resize(mask[k])
            imsave(f"{processedpath}/image{k}/{i}.tif", np_im)
            imsave(f"{processedpath}/label{k}/{i}.tif", mk.astype('int'))

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
    for i in range(8):
        os.mkdir(f"{trainpath}/image{i}")
        os.mkdir(f"{trainpath}/label{i}")
        os.mkdir(f"{testpath}/image{i}")
        os.mkdir(f"{testpath}/label{i}")

    idxs = os.listdir(f"{datapath}/image3/")
    train_idxs, test_idxs = train_test_split(idxs, test_size=.2)
    for i, idx in enumerate(train_idxs):
        for j in range(8):
            im_source = f"{datapath}/image{j}/{idx}"
            im_dest = f"{trainpath}/image{j}/{i}.tif"
            shutil.copyfile(im_source, im_dest)

            msk_source = f"{datapath}/label{j}/{idx}"
            msk_dest = f"{trainpath}/label{j}/{i}.tif"
            shutil.copy(msk_source, msk_dest)

    for i, idx in enumerate(test_idxs):
        for j in range(8):
            im_source = f"{datapath}/image{j}/{idx}"
            im_dest = f"{testpath}/image{j}/{i}.tif"
            shutil.copyfile(im_source, im_dest)

            msk_source = f"{datapath}/label{j}/{idx}"
            msk_dest = f"{testpath}/label{j}/{i}.tif"
            shutil.copy(msk_source, msk_dest)


if __name__ == "__main__":
    pass
