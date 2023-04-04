import tensorflow as tf
from tensorflow.keras import layers

def densenet201():
    base_model = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    outputs = base_model.output
    outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-4))(outputs)
    outputs = tf.keras.layers.Dropout(0.5)(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(outputs)
    outputs = tf.keras.layers.Dropout(0.3)(outputs)
    predicts = layers.Dense(2, activation="softmax")(outputs)

    final_model = tf.keras.Model(inputs=base_model.input, outputs=predicts, name='densenet201')
    return final_model

def vgg16():
    base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    outputs = base_model.output
    outputs = layers.AveragePooling2D((2, 2), strides=(2, 2))(outputs)
    outputs = layers.Flatten(name="flatten")(outputs)
    outputs = layers.Dense(1024, activation="relu")(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.Dense(512, activation="relu")(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.Dense(64, activation="relu")(outputs)
    predicts = layers.Dense(2, activation="softmax")(outputs)

    final_model = tf.keras.Model(inputs=base_model.input, outputs=predicts, name='vgg16')
    return final_model

def vgg19():
    base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    outputs = base_model.output
    outputs = layers.AveragePooling2D((2, 2), strides=(2, 2))(outputs)
    outputs = layers.Flatten(name="flatten")(outputs)
    outputs = layers.Dense(1024, activation="relu")(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.Dense(512, activation="relu")(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.Dense(64, activation="relu")(outputs)
    predicts = layers.Dense(2, activation="softmax")(outputs)

    final_model = tf.keras.Model(inputs=base_model.input, outputs=predicts, name='vgg19')
    return final_model

def xception():
    base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    outputs = base_model.output
    outputs = layers.experimental.preprocessing.RandomRotation((-0.15, 0.15))(outputs)
    outputs = layers.experimental.preprocessing.RandomFlip('horizontal')(outputs)
    outputs = layers.GlobalMaxPooling2D()(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(units=512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.L1L2())(outputs)
    outputs = tf.keras.layers.GaussianDropout(0.5)(outputs)
    outputs = tf.keras.layers.Dense(units=128, activation='relu')(outputs)
    outputs = tf.keras.layers.GaussianDropout(0.5)(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    predicts = layers.Dense(2, activation="softmax")(outputs)

    final_model = tf.keras.Model(inputs=base_model.input, outputs=predicts, name='xception')
    return final_model

def inceptionv3():
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    outputs = base_model.output
    outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(312, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(l1=2e-5, l2=1e-4))(outputs)
    outputs = tf.keras.layers.Dropout(0.5)(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(outputs)
    outputs = tf.keras.layers.Dropout(0.3)(outputs)
    predicts = layers.Dense(2, activation="softmax")(outputs)

    final_model = tf.keras.Model(inputs=base_model.input, outputs=predicts, name='inceptionv3')
    return final_model

def mobilenetv2():
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    outputs = base_model.output
    outputs = layers.AveragePooling2D((2, 2), strides=(2, 2))(outputs)
    outputs = layers.Flatten(name="flatten")(outputs)
    outputs = layers.Dense(1024, activation="relu")(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.Dense(512, activation="relu")(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.Dense(64, activation="relu")(outputs)
    predicts = layers.Dense(2, activation="softmax")(outputs)

    final_model = tf.keras.Model(inputs=base_model.input, outputs=predicts, name='mobilenetv2')
    return final_model

def resnet50v2():
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    outputs = base_model.output
    outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-4))(outputs)
    outputs = tf.keras.layers.Dropout(0.5)(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(outputs)
    outputs = tf.keras.layers.Dropout(0.3)(outputs)
    predicts = layers.Dense(2, activation="softmax")(outputs)

    final_model = tf.keras.Model(inputs=base_model.input, outputs=predicts, name='resnet50v2')
    return final_model