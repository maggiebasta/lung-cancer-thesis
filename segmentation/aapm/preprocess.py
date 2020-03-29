import os
import sys

import numpy as np
from PIL import Image, ImageEnhance
from skimage.io import imread, imsave

sys.path.append("../../")
from preprocess_helpers import get_lung_mask, normalize, resize

# Images that need to have lung fields 'partially' manually segmented
special_imgs = [
    '17.tif', '19.tif', '27.tif',
    '32.tif', '40.tif', '51.tif',
    '6.tif', '69.tif'
]


def preprocess_img(img, special=False, manual_lung_mask=None):
    lung_mask = get_lung_mask(img).astype('float')
    if special:
        lung_mask += manual_lung_mask
        lung_mask = np.clip(lung_mask, 0, 1)

    lung_mask = resize(lung_mask)
    img = normalize(img)
    img = resize(img)
    img = img*lung_mask
    pil_im = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(pil_im)
    enhanced_im = enhancer.enhance(2.0)
    return np.array(enhanced_im)


def preprocess(extracted_path, processed_path):
    os.mkdir(processed_path)
    img_paths = os.listdir(extracted_path)
    n = len(img_paths)
    for i, img_path in enumerate(img_paths):
        sys.stdout.write(f"\rProcessing...{i+1}/{n}")
        sys.stdout.flush()
        img = imread(extracted_path + '/' + img_path)
        if img_path in special_imgs:
            lung_mask = imread(
                f"data/special_masks/{img_path}"
            ).astype('float')
            processed_img = preprocess_img(
                img, special=True, manual_lung_mask=lung_mask)
        else:
            processed_img = preprocess_img(img)
        imsave(f"{processed_path}/{img_path}", processed_img)
    print(f"\nComplete.")
