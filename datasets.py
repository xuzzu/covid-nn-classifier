from import_statements import *


""" DATA FOLDERS WITH CLASSES MUST BE NAMED AS THE CLASSES AND LIE IN "train" FOLDER """

# Data pipeline
def make_generator():
    TRAIN_DATASET_PATH = fr'{ROOT_DATASETS_PATH}\{DATASET_NAME}\train'
    # VAL_DATASET_PATH = fr'{ROOT_DATASETS_PATH}\{DATASET_NAME}\val'
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(directory=TRAIN_DATASET_PATH,
                                                                        shuffle=True,
                                                                        validation_split=SPLIT,
                                                                        image_size=IMG_SIZE,
                                                                        color_mode="rgb", label_mode="categorical",
                                                                        seed=15,
                                                                        subset="training", batch_size=BATCH_SIZE)

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(directory=TRAIN_DATASET_PATH,
                                                                      shuffle=True,
                                                                      validation_split=SPLIT,
                                                                      image_size=IMG_SIZE,
                                                                      color_mode="rgb", label_mode="categorical",
                                                                      seed=15,
                                                                      subset="validation", batch_size=BATCH_SIZE)

    DATASET_NAME = ''
    return (train_dataset, val_dataset)

def make_test_generator(dataset_name):
    DATASET_NAME = dataset_name
    TEST_DATASET_PATH = fr'{ROOT_DATASETS_PATH}/{DATASET_NAME}/test'
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(directory=TEST_DATASET_PATH,
                                                                       shuffle=True,
                                                                       image_size=IMG_SIZE,
                                                                       color_mode="rgb", label_mode="categorical",
                                                                       batch_size=BATCH_SIZE)
    return test_dataset