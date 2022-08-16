# Covid NN Classifier

One of the backbones of medical imaging research. Does binary classification on CT scans of lungs (Covid/Non-Covid) using transfer learning approach. Has a ready-to-use data pipeline and allows testing of different models in a single run. Incorporates 3D modelling of DICOM files for further work on 3D segmentation.

## Main file: all_models.ipynb

Has a number of popular models for transfer learning (mobilentev2, resnet50, etc.) with averagely optimal meta-layers. The pipeline allows direct import of png/jpeg images from nested directories.

### Work in progress:
- Data Cleaning (segmentation of areas of interest)
- Fine-tuning adjustments
- Reading of DICOM files

## Main file: 3D Plot.ipynb

Reads DICOM files from specified directory and does basic data augmentation to highlight lung tissues and plot them in 3D. 

### Work in progress:
- Segmentation of only lungs-related tissues
- Optimization of DICOM import
- Change of 3D visualization tool
