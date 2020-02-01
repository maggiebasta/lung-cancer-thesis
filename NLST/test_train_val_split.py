import os

import numpy as np
import pandas as pd
import pickle as pkl
from skimage.io import imsave
from sklearn.model_selection import train_test_split
from PIL import Image


def ttv_split(
    processed_path='data/nlst_processed',
    roi_3d_path='data/nlst_rois_3d',
    table_path='data/nlst_table_cleaned.csv',
):
    """
    Splits processed images into test, train and validation sets. Saves
    indiviudal nodules and tracks which nodules belong to which patient
    in a saved lookup table
    """

    # get table for labels to stratify
    df_nlst = pd.read_csv(table_path)
    df_recurr = df_nlst[['pid', 'recurrence']].drop_duplicates()

    # track labels for patients
    RecurTable = {
        pid: int(recurr) for _, (pid, recurr) in df_recurr.iterrows()
    }

    # partition into train, test and validation sets (stratify by recurrence)
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

    # construct directory structure
    for d in ['nlst_train', 'nlst_val', 'nlst_test']:
        os.mkdir(f'data/{d}')
        os.mkdir(f'data/{d}/image_roi_3d')
        os.mkdir(f'data/{d}/image_roi_3d/label')
        os.mkdir(f'data/{d}/image_roi_3d/pkls')
        for i in range(50):
            os.mkdir(f'data/{d}/image_roi_3d/{i}')

    # copy processed files into directories for train, test, and val sets
    dirs_and_ids = [
        ('nlst_train', train_ids),
        ('nlst_val', val_ids),
        ('nlst_test', test_ids)
    ]
    for d, pids in dirs_and_ids:
        count = 0
        PatientLookup = {}
        for pid in pids:
            pid_roi_path = f'{roi_3d_path}/{pid}'
            label = RecurTable[int(pid)]
            if not os.path.exists(pid_roi_path):
                print(f'No ROIs for {pid}')
            else:
                for im in [f[:-4] for f in os.listdir(pid_roi_path)]:
                    with open(f'{pid_roi_path}/{im}.pkl', 'rb') as input_file:
                        cube = pkl.load(input_file)
                    for i, slc in enumerate(cube[:50]):
                        slc = np.array(Image.fromarray(slc))
                        imsave(
                            f'data/{d}/image_roi_3d/{i}/{count}.tif',
                            slc
                        )
                    pkl.dump(
                        (cube, label),
                        open(f'data/{d}/image_roi_3d/pkls/{count}.pkl', "wb")
                    )
                    imsave(
                        f'data/{d}/image_roi_3d/label/{count}.tif',
                        np.array(Image.fromarray(np.array([[label]]), 'L'))
                    )
                    PatientLookup[count] = pid
                    count += 1
        pkl.dump(
            PatientLookup,
            open(f'data/{d}/image_roi_3d/patient_lookup.pkl', "wb")
        )
