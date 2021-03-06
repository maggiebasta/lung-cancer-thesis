# U-Net 3D

## Pipeline
1. Download, extract and format data from S3 (call to data_downloader.py or run cells in downloader.ipynb) 
2. Format data into generators for training (call data_generator.py or run initial cells in trainUnet.ipynb) 
3. Train and Test the U-Net (run cells in trainUnet.ipynb) 

## Files

### Python files
- data_downloader.py: downloads, extracts and formats data from S3
- data_generator.py: implements Keras Image generators for U-Net
- dice_loss.py: implements dice coefficient loss function for Keras models
- lidc_helpers.py: helper functions for EDA and data extraction
- unet.py: implementation of the U-Net architecture 


### Core Notebooks
- downloader.ipynb: uses data_downloader.py to download, extract and format data from S3 via 
- trainUnet.ipynb: instantiates data generators and the U-Net, trains and tests the U-Net



