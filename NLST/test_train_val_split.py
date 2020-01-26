import os
import sys

import numpy as np
import pandas as pd
import pickle as pkl
from skimage.io import imread, imsave
from sklearn.model_selection import train_test_split
from PIL import Image


def ttv_split(
    processed_path='data/nlst_processed',
    roi_2d_path='data/nlst_rois_2d',
    roi_3d_path='data/nlst_rois_3d',
    table_path='data/nlst_table_cleaned.csv',
):
    df_nlst = pd.read_csv(table_path)
    df_recurr = df_nlst[['pid', 'recurrence']].drop_duplicates()

    RecurTable = {pid: recurr for _, (pid, recurr) in df_recurr.iterrows()}

    train_ids, test_ids, train_recur, _ = train_test_split(
        list(df_recurr.pid),
        list(df_recurr.recurrence),
        stratify=df_recurr.recurrence,
        test_size=0.2,
        random_state=11
    )

    train_ids, val_ids, _, _ = train_test_split(
        train_ids,
        train_recur,
        stratify=train_recur,
        test_size=0.1,
        random_state=26
    )

    os.mkdir('data/nlst_train')
    os.mkdir('data/nlst_train/image_full')
    os.mkdir('data/nlst_train/image_full/1')
    os.mkdir('data/nlst_train/image_full/0')
    os.mkdir('data/nlst_train/image_roi_2d')
    os.mkdir('data/nlst_train/image_roi_2d/1')
    os.mkdir('data/nlst_train/image_roi_2d/0')
    os.mkdir('data/nlst_train/image_roi_3d')

    for i in range(50):
        os.mkdir(f'data/nlst_train/image_roi_3d/{i}')
    os.mkdir('data/nlst_train/image_roi_3d/label')

    os.mkdir('data/nlst_val')
    os.mkdir('data/nlst_val/image_full')
    os.mkdir('data/nlst_val/image_full/1')
    os.mkdir('data/nlst_val/image_full/0')
    os.mkdir('data/nlst_val/image_roi_2d')
    os.mkdir('data/nlst_val/image_roi_2d/1')
    os.mkdir('data/nlst_val/image_roi_2d/0')
    os.mkdir('data/nlst_val/image_roi_3d')
    for i in range(50):
        os.mkdir(f'data/nlst_val/image_roi_3d/{i}')
    os.mkdir('data/nlst_val/image_roi_3d/label')

    os.mkdir('data/nlst_test')
    os.mkdir('data/nlst_test/image_full')
    os.mkdir('data/nlst_test/image_full/1')
    os.mkdir('data/nlst_test/image_full/0')
    os.mkdir('data/nlst_test/image_roi_2d')
    os.mkdir('data/nlst_test/image_roi_2d/1')
    os.mkdir('data/nlst_test/image_roi_2d/0')
    os.mkdir('data/nlst_test/image_roi_3d')
    os.mkdir('data/nlst_test/image_roi_3d/pkls')
    for i in range(50):
        os.mkdir(f'data/nlst_test/image_roi_3d/{i}')
    os.mkdir('data/nlst_test/image_roi_3d/label')

    count = 0
    for pid in val_ids:
        pid_full_path = f'{processed_path}/{pid}'
        pid_roi_2d_path = f'{roi_2d_path}/{pid}'
        pid_roi_3d_path = f'{roi_3d_path}/{pid}'
        label = RecurTable[int(pid)]
        try:
            for im in [f[:-4] for f in os.listdir(pid_roi_3d_path)]:
                imsave(
                    f'data/nlst_val/image_full/{label}/{count}.tif',
                    imread(f'{pid_full_path}/{im}.tif')
                )
                imsave(
                    f'data/nlst_val/image_roi_2d/{label}/{count}.tif',
                    imread(f'{pid_roi_2d_path}/{im}.tif')
                )
                with open(f'{pid_roi_3d_path}/{im}.pkl', 'rb') as input_file:
                    cube = pkl.load(input_file)
                for i, slc in enumerate(cube[:50]):
                    slc = np.array(Image.fromarray(slc))
                    imsave(
                        f'data/nlst_val/image_roi_3d/{i}/{count}.tif',
                        slc
                    )
                imsave(
                    f'data/nlst_val/image_roi_3d/label/{count}.tif',
                    np.array(Image.fromarray(np.array([[label]]), 'L'))
                )
                count += 1
        except FileNotFoundError:
            print(f'No ROIs for {pid}')

    count = 0
    for pid in train_ids:
        pid_full_path = f'{processed_path}/{pid}'
        pid_roi_2d_path = f'{roi_2d_path}/{pid}'
        pid_roi_3d_path = f'{roi_3d_path}/{pid}'
        label = RecurTable[int(pid)]
        try:
            for im in [f[:-4] for f in os.listdir(pid_roi_3d_path)]:
                imsave(
                    f'data/nlst_train/image_full/{label}/{count}.tif',
                    imread(f'{pid_full_path}/{im}.tif')
                )
                imsave(
                    f'data/nlst_train/image_roi_2d/{label}/{count}.tif',
                    imread(f'{pid_roi_2d_path}/{im}.tif')
                )
                with open(f'{pid_roi_3d_path}/{im}.pkl', 'rb') as input_file:
                    cube = pkl.load(input_file)
                for i, slc in enumerate(cube[:50]):
                    slc = np.array(Image.fromarray(slc))
                    imsave(
                        f'data/nlst_train/image_roi_3d/{i}/{count}.tif',
                        slc
                    )
                imsave(
                    f'data/nlst_train/image_roi_3d/label/{count}.tif',
                    np.array(Image.fromarray(np.array([[label]]), 'L'))
                )
                count += 1
        except FileNotFoundError:
            print(f'No ROIs for {pid}')

    count = 0
    PatientLookup = {}
    for pid in test_ids:
        pid_full_path = f'{processed_path}/{pid}'
        pid_roi_2d_path = f'{roi_2d_path}/{pid}'
        pid_roi_3d_path = f'{roi_3d_path}/{pid}'
        label = RecurTable[int(pid)]
        try:
            for im in [f[:-4] for f in os.listdir(pid_roi_3d_path)]:
                imsave(
                    f'data/nlst_test/image_full/{label}/{count}.tif',
                    imread(f'{pid_full_path}/{im}.tif')
                )
                imsave(
                    f'data/nlst_test/image_roi_2d/{label}/{count}.tif',
                    imread(f'{pid_roi_2d_path}/{im}.tif')
                )
                with open(f'{pid_roi_3d_path}/{im}.pkl', 'rb') as input_file:
                    cube = pkl.load(input_file)
                pkl.dump(
                    (cube, label),
                    open(f'data/nlst_test/image_roi_3d/pkls/{count}.pkl', "wb")
                )
                for i, slc in enumerate(cube[:50]):
                    slc = np.array(Image.fromarray(slc))
                    imsave(
                        f'data/nlst_test/image_roi_3d/{i}/{count}.tif',
                        slc
                    )
                imsave(
                    f'data/nlst_test/image_roi_3d/label/{count}.tif',
                    np.array(Image.fromarray(np.array([[label]]), 'L'))
                )
                PatientLookup[count] = pid
                count += 1
        except FileNotFoundError:
            print(f'No ROIs for {pid}')
    pkl.dump(
        PatientLookup,
        open(f'data/nlst_test/image_roi_3d/patient_lookup.pkl', "wb")
    )
