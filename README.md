# Convolutional Neural Networks for the Automated Segmentation and Recurrence Risk Prediction of Surgically Resected Lung Tumors

## Overview 
Welcome to the repository for my undergraduate senior thesis - *Convolutional Neural Networks for the Automated Segmentation and Recurrence Risk Prediction of Surgically Resected Lung Tumors*. This repository contains all of the code used to implement the models and experiments discussed in the thesis. The pipeline is illustrated below. 

## Pipeline
<p align="center">
  <img src="https://github.com/maggiebasta/lung-cancer-thesis/blob/master/figures/system_overview.jpg?raw=true" width="550">
</p>


## Repository Structure

### Preprocessing
The file [preprocess_helpers.py](https://github.com/maggiebasta/lung-cancer-thesis/blob/master/preprocess_helpers.py) contains the functions used in the preprocessing pipeline. These functions are called during the extraction/preprocessing from each dataset used in both segmentation and recurrence prediction. 

### Segmentation 
All code for the processing, training, testing, and experiments of the segmentation stage are included in the [segmentation](https://github.com/maggiebasta/lung-cancer-thesis/tree/master/segmentation) folder of this repo. An additional README is included within the folder for further details.

### Recurrence Prediction
All code for the processing, training, testing, and experiments of the recurrence prediction stage are included in the [recurrence_prediction](https://github.com/maggiebasta/lung-cancer-thesis/tree/master/recurrence_prediction) folder of this repo. An additional README is included within the folder for further details.

## Data
The project requires a significant amount of external data to run. Not all of this data is publically available. However, below are the links to the available information for each dataset. 
- [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) (publically available)
- [LUNA16](https://luna16.grand-challenge.org/Home/) (publically available)
- [SPIE-AAPM](https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM+Lung+CT+Challenge) (publically available)
- [NLST](https://cdas.cancer.gov/nlst/): (not publically available, interested proejcts can submit requests)

The expected directory structure is provided in the README of each folder. 

## Dependencies
The project relies on the following dependencies: 
1. [Matplotlib](https://matplotlib.org/)
2. [Numpy](https://numpy.org/)
3. [OpenCv](https://opencv.org/)
4. [Pandas](https://pandas.pydata.org/)
5. [Pickle](https://docs.python.org/3/library/pickle.html)
6. [PIL](https://www.pythonware.com/products/pil/)
7. [Pydicom](https://pydicom.github.io/)
8. [Scikit-image](https://scikit-image.org/docs/dev/api/skimage.html)
9. [Scikit-learn](https://scikit-learn.org/stable/)
10. [Tensorflow](https://www.tensorflow.org/) (version 1.14.0 used)

*Note: this project was implemented using a GPU optimized AWS EC2 instance. A local machine will most likely not be able to run (the majority of) the code in this repo.*
