import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from skimage.io import imread, imsave

sys.path.append('../')
from preprocess_helpers import get_lung_mask, normalize, resize

# Images that need to have lung fields 'partially' manually segmented
special_imgs_1 = [
    '100242_T0_1.tif', '103359_T0_4.tif', '104564_T0_2.tif',
    '105012_T1_1.tif', '105771_T2_3.tif', '106204_T2_2.tif',
    '107002_T1_1.tif', '107002_T1_2.tif', '107791_T0_1.tif',
    '111859_T0_1.tif', '117799_T0_1.tif', '119597_T0_1.tif',
    '122376_T0_3.tif', '122376_T1_3.tif', '122376_T2_3.tif',
    '128156_T0_1.tif', '130162_T0_1.tif', '130162_T0_3.tif',
    '132227_T0_1.tif', '200221_T1_1.tif', '201528_T1_1.tif',
    '201528_T2_6.tif', '201938_T0_2.tif', '202014_T0_1.tif',
    '204711_T1_1.tif', '206354_T1_1.tif', '206359_T1_4.tif',
    '206737_T2_1.tif', '206737_T2_4.tif', '207206_T0_1.tif',
    '207206_T0_4.tif', '207954_T0_2.tif', '207954_T0_3.tif',
    '207954_T0_4.tif', '104999_T0_2.tif', '104999_T1_1.tif',
    '113814_T0_2.tif', '113814_T0_3.tif', '117950_T1_2.tif',
    '117950_T1_3.tif', '117950_T1_4.tif', '117950_T2_3.tif',
    '119358_T0_1.tif', '119358_T1_2.tif', '119911_T2_1.tif',
    '205714_T2_2.tif', '210090_T2_5.tif', '214923_T0_3.tif',
    '112956_T0_1.tif', '121099_T0_1.tif', '207782_T0_2.tif',
    '207782_T1_2.tif', '207782_T1_5.tif'
]

# Images that need to have lung fields 'completely' manually segmented
special_imgs_2 = [
    '100954_T0_3.tif', '105340_T0_1.tif', '112506_T2_6.tif',
    '104778_T0_4.tif', '118297_T1_2.tif', '118297_T1_3.tif'
]


def preprocess_img(img, special=0, manual_lung_mask=None):
    lung_mask = get_lung_mask(img).astype('float')
    if special == 1:
        lung_mask += manual_lung_mask
        lung_mask = np.clip(lung_mask, 0, 1)
    if special == 2:
        lung_mask = manual_lung_mask

    lung_mask = resize(lung_mask)
    img = normalize(img)
    img = resize(img)
    img = img*lung_mask
    pil_im = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(pil_im)
    enhanced_im = enhancer.enhance(2.0)
    return np.array(enhanced_im)


def preprocess(extracted_path, processed_path, table_path='data/nlst_table_cleaned.csv'):
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
                    img, special=1, manual_lung_mask=lung_mask)
            elif str(pid) + '_' + im_path in special_imgs_2:
                lung_mask = imread(
                    f'data/nlst_special_masks/{pid}_{im_path}'
                ).astype('float')
                processed_img = preprocess_img(
                    img, special=2, manual_lung_mask=lung_mask)
            else:
                processed_img = preprocess_img(img)
            imsave(f'{processed_path}/{pid}/{im_path}', processed_img)
    print(f'\nComplete.')

