from model_functions import *
from plot_functions import plot_training
from datasets import make_test_generator
from import_statements import *

def evaluate_model(model, dataset):
    train_datagen, val_datagen = make_test_generator(dataset)
    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='prec'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.TruePositives(name='tp'),
                           tf.keras.metrics.TrueNegatives(name='tn'),
                           tf.keras.metrics.FalsePositives(name='fp'),
                           tf.keras.metrics.FalseNegatives(name='fn')])
    history = model.fit(train_datagen,
                        epochs=NUM_EPOCHS,
                        validation_data=val_datagen,
                        verbose=2,
                        callbacks=[ACC_CALLBACK])
    model_name = model.name
    plot_training(history, model_name, dataset)
