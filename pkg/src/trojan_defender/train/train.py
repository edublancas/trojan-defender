"""
Training a CNN for image recognition

Based on: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
"""
import logging
import numpy as np
import keras
from trojan_defender import log
from trojan_defender.datasets import Dataset


def cnn(dataset, model_loader, batch_size=128, epochs=12):
    logger = logging.getLogger(__name__)

    np.random.seed(0)

    model = model_loader(dataset.input_shape, dataset.num_classes)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(dataset.x_train, dataset.y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(dataset.x_test, dataset.y_test))

    score = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)

    logger.info('Test loss: %.2f', score[0])
    logger.info('Test accuracy: %.2f', score[1])

    # log.experiment(model)

    return model
