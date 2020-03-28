import os
import sys

import numpy as np
import pandas as pd
import pickle as pkl
from PIL import Image, ImageEnhance
from skimage.io import imread, imsave

sys.path.append('../')
from preprocess_helpers import get_lung_mask, normalize, resize

TABLE_PATH = 'data/nlst_table_cleaned.csv'

# Images with difficult "custom" lung fields
special_imgs_1, special_imgs_2 = pkl.load(open('data/special_ids.pkl', 'rb'))


def preprocess_img(img, special=0, custom_lung_mask=None):
    """
    Preprocessing pipeline for individual images. Applies lung field
    segmentation, resizes, normalizes and enhances constrast

    :param img: input image
    :param special: whether or not the image has a difficult/custom lung mask
    :param custom_lung_mask: custom lung mask (if special != 0)
    """
    lung_mask = get_lung_mask(img).astype('float')
    if special == 1:
        lung_mask += custom_lung_mask
        lung_mask = np.clip(lung_mask, 0, 1)
    if special == 2:
        lung_mask = custom_lung_mask

    lung_mask = resize(lung_mask)
    img = normalize(img)
    img = resize(img)
    img = img*lung_mask
    pil_im = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(pil_im)
    enhanced_im = enhancer.enhance(2.0)
    return np.array(enhanced_im)


def preprocess(extracted_path, processed_path, table_path=TABLE_PATH):
    """
    Preprocesses all extracted slices using the routine in preprocess_img()

    :param extracted_path: path where extracted slices are
    :param processed_path: where to save processed images
    :param table_path: path to NLST table (for label data)
    """
    os.mkdir(processed_path)
    pids = pd.read_csv(table_path).pid.unique()
    n = len(pids)
    for i, pid in enumerate(pids):
        sys.stdout.write(f'\rProcessing...{i+1}/{n}')
        sys.stdout.flush()
        os.mkdir(processed_path + '/' + str(pid))
        for im_path in os.listdir(extracted_path + '/' + str(pid)):
            img = imread(
                os.path.join(extracted_path + '/' + str(pid), im_path))
            if str(pid) + '_' + im_path in special_imgs_1:
                lung_mask = imread(
                    f'data/nlst_special_masks/{pid}_{im_path}'
                ).astype('float')
                processed_img = preprocess_img(
                    img, special=1, custom_lung_mask=lung_mask)
            elif str(pid) + '_' + im_path in special_imgs_2:
                lung_mask = imread(
                    f'data/nlst_special_masks/{pid}_{im_path}'
                ).astype('float')
                processed_img = preprocess_img(
                    img, special=2, custom_lung_mask=lung_mask)
            else:
                processed_img = preprocess_img(img)
            imsave(f'{processed_path}/{pid}/{im_path}', processed_img)
    print(f'\nComplete.')
