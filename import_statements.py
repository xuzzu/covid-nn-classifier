import tensorflow as tf
import plotly.graph_objects as go
import json
from tensorflow.keras import layers
import cv2
import numpy as np
from PIL import Image
import tensorflow.keras as K
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
from callbacks import MyThresholdCallback

""" CONSTANTS AND PARAMETERS """

NUM_EPOCHS = 100
BATCH_SIZE = 32
IMG_SIZE = (256, 256)
SPLIT = 0.2
ROOT_DATASETS_PATH = r''
DATASET_NAME = ''
ACCURACY_THRESHOLD = 0.99
ACC_CALLBACK = MyThresholdCallback(ACCURACY_THRESHOLD)
DATASETS_LIST = ['CovidX', 'SaoPaulo', 'UCSD'] # Name of the datasets folders in your root folder

