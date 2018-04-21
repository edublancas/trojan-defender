"""
Training a CNN for image classification
"""
import logging
import numpy as np
import keras


def mnist_cnn(dataset, model_loader, batch_size=128, epochs=12,
              deterministic=True, verbose=True):
    """
    Taken from:
    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    """
    logger = logging.getLogger(__name__)

    if deterministic:
        np.random.seed(0)

    model = model_loader(dataset.input_shape, dataset.num_classes)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    logger.info('Fitting model...')

    model.fit(dataset.x_train, dataset.y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              validation_data=(dataset.x_test, dataset.y_test))

    score = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)

    logger.info('Test loss: %.2f', score[0])
    logger.info('Test accuracy: %.2f', score[1])

    return model


def cifar10_cnn(dataset, model_loader, batch_size=32, epochs=100,
                deterministic=True, verbose=True):
    """
    Taken from:
    https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
    """
    logger = logging.getLogger(__name__)

    if deterministic:
        np.random.seed(0)

    model = model_loader(dataset.input_shape, dataset.num_classes)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    logger.info('Fitting model...')

    model.fit(dataset.x_train, dataset.y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(dataset.x_test, dataset.y_test),
              shuffle=True,
              verbose=verbose)

    score = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)

    logger.info('Test loss: %.2f', score[0])
    logger.info('Test accuracy: %.2f', score[1])

    return model
