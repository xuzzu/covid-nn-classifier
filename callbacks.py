import tensorflow as tf

class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_accuracy"]
        train_acc = logs["accuracy"]
        if val_acc >= self.threshold and train_acc >= self.threshold:
            self.model.stop_training = True