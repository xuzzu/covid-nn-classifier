from import_statements import *
from plot_functions import plot_training
from model_functions import *
from datasets import make_generator
from evaluation import evaluate_model

def main(eval=False):
    models_list = [densenet201(), vgg16(), vgg19(), xception(), inceptionv3(), mobilenetv2(), resnet50v2()]
    final_pred = 0.0
    i = 0

    for dataset in DATASETS_LIST:
        train_datagen, val_datagen = make_generator(dataset)
        for model in models_list:
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

            # If we want to evaluate the model
            if eval:
                evaluate_model(model, dataset)


if __name__ == '__main__':
    main()
