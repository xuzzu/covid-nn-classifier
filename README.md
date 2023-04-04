# CNN Covid Classifier
This code implements image classification using popular pre-trained models (DenseNet201, VGG16, etc.) on a binary classification task. The dataset is expected to be divided into 'train' and 'test' folders, and the subfolders' names must match the classes' names.

It was specifically used and developed for X-Ray and CT scan classification, but can be used for other tasks as well with proper readjustment of meta-layers in models’ structures.

## Prerequisites
•	Python 3

•	TensorFlow 2.x

•	plotly

•	opencv-python

•	numpy

•	Pillow

•	matplotlib

•	scikit-image

## How to use
1.	Clone the repository.
2.	Make sure you have all the dependencies installed.
3.	Modify the necessary parameters in import_statements.py.
4.	Run the code python main.py.
## Additional Notes
•	The necessary parameters for the models can be adjusted in import_statements.py.

•	The TensorFlow generators require folders' names as class names, and they must lie in the 'train' or 'test' folder.

### Callback
The script contains a custom callback class MyThresholdCallback that stops the training when the validation and training accuracy reaches the threshold.
### Models
•	DenseNet201

•	VGG16

•	VGG19

•	Xception

•	InceptionV3

•	MobileNetV2

•	ResNet50V2

### Plotting
The code also includes a plot_training function that generates a line plot of the training accuracy over each epoch. The function saves an HTML file that contains the plot to the root folder.
