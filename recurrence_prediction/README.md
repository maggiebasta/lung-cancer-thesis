# Recurrent Prediction Stage


This folder contains the core of recurrence prediction implementation. It contains all processing of NLST images and tabular data, the implementation of the architecture, and the training and testing of the architecutre on NLST.

## Pipeline:

1. [nlst_tabular_processing.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/recurrence_prediction/nlst_tabular_processing.ipynb) runs all necessary processing scripts to process tabular data
2. [nlst_image_processing.ipynb.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/recurrence_prediction/nlst_image_processing.ipynb) runs all necessary processing scripts to process imaging data
3. [nlst_train.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/recurrence_prediction/nlst_train.ipynb) implements, trains and evaluates the network and the random forest models.

## Expected directory structure 
The directory structure expected to run this code, *including datasets* is provided below. 
```
├── recurrence_prediction
│   ├── data
│   │   ├── geometric_data.pkl
│   │   ├── nlst_567
│   │   │   ├── nlst_567_ct_ab_20191108.csv
│   │   │   ├── nlst_567_ct_image_info_20191108.csv
│   │   │   ├── nlst_567_pid_list_20191108.csv
│   │   │   ├── nlst_567_prsn_20191108.csv
│   │   │   └── treatment.data.d100517.csv
│   │   ├── nlst_extracted
│   │   │   ├── ...
│   │   ├── nlst_extracted_3d
│   │   │   ├── ...
│   ├── data_generator_3d.py
│   ├── model.py
│   ├── nlst_image_processing.ipynb
│   ├── nlst_tabular_processing.ipynb
│   ├── nlst_train.ipynb
│   ├── plotting_helpers.py
│   ├── preprocess.py
│   ├── random_forest.py
│   ├── roi_predict.py
│   └── test_train_val_split.py

```
