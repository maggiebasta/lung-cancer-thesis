# Recurrent Prediction Stage


This folder contains the core of recurrence prediction implementation. It contains all processing of NLST images and tabular data, the implementation of the architecture, and the training and testing of the architecutre on NLST.

**Pipeline:**

1. [nlst_tabular_processing.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/recurrence_prediction/nlst_tabular_processing.ipynb) runs all necessary processing scripts to process tabular data
2. [nlst_image_processing.ipynb.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/recurrence_prediction/nlst_image_processing.ipynb) runs all necessary processing scripts to process imaging data
3. [nlst_train.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/recurrence_prediction/nlst_train.ipynb) implements, trains and evaluates the network and the random forest models.
