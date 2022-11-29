**Pipeline:**

1. [preprocess_notebook.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/segmentation/unet_2d/preprocess_notebook.ipynb) runs all necessary preprocessing scripts
2. [trainUnet.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/segmentation/unet_2d/trainUnet.ipynb) initializes and trains the segmentation network.
3. [testUnet.ipynb](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/segmentation/unet_2d/testUnet.ipynb) evaluates the trained network on the test set


## Expected directory structure
The directory structure expected to run this code, *including datasets* is provided below.
```
segmentation
    ├── data
    │   ├── raw_data
    │   │   └── LIDC-IDRI
    │   │       ├── LIDC-IDRI-0001
    │   │       ├── LIDC-IDRI-0002
    |   |       ├── ...
    ├── data_generator.py
    ├── helpers.py
    ├── metrics.py
    ├── model.py
    ├── preprocess.py
    ├── preprocess_notebook.ipynb
    ├── testUnet.ipynb
    └── trainUnet.ipynb

```