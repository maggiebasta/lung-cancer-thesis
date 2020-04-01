# Segmentation Stage

## unet_2D

This folder contains the core of segmentation implementation. It contains all processing of LIDC-IDRI images, the implementation of the architecture, and the training and testing of the architecutre on LIDC-IDRI.

**Pipeline:**

1. [preprocess_notebook.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/segmentation/unet_2d/preprocess_notebook.ipynb) runs all necessary preprocessing scripts
2. [trainUnet.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/segmentation/unet_2d/trainUnet.ipynb) initializes and trains the segmentation network.
3. [testUnet.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/segmentation/unet_2d/testUnet.ipynb) evaluates the trained network on the test set

## unet_3D

This folder is a work in progress. It contains initial exploratory work into a 3D architecture, but is **incomplete**. 

## aapm

This folder contains the code used to run the tests on the SPIE-AAPM dataset. 
**Pipeline:**

1. [aapm_processing.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/segmentation/aapm/aapm_processing.ipynb) runs all necessary preprocessing scripts
2. [aapm_test.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/segmentation/aapm/aapm_test.ipynb) tests the trained segmentation network on the preprocessed SPIE-AAPM images.


## Expected directory structure
The directory structure expected to run this code, including datasets is provided below.
```
segmentation
    ├── aapm
    │   ├── aapm_processing.ipynb
    │   ├── aapm_test.ipynb
    │   ├── data
    │   │   ├── CalibrationSet_NoduleData.xlsx
    │   │   ├── raw_data
    │   │   │ 	 ├──...				
    │   │   └── TestSet_NoduleData_PublicRelease_wTruth.xlsx
    │   ├── predict.py
    │   ├── preprocess.py
    ├── metrics.py
    ├── raw_data
    │   └── LIDC-IDRI
    │       ├──...
    ├── unet_2d
    │   ├── data
    │   │   ├── ...
    │   ├── data_generator.py
    │   ├── helpers.py
    │   ├── model.py
    │   ├── other_models.py
    │   ├── preprocess_notebook.ipynb
    │   ├── preprocess.py
    │   ├── testUnet.ipynb
    │   ├── trainUnet.ipynb
    │   └── uids.pkl
    └── unet_3d
        ├── ...
```
