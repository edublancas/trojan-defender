"""
Training a CNN for image classification
"""
import logging
import numpy as np
import keras


def train(dataset, model_loader, batch_size=128, epochs=12,
          deterministic=True, verbose=True):
    """
    Taken from:
    https://github.com/keras-team/keras/blob/master/examples/
    """
    logger = logging.getLogger(__name__)

    if deterministic:
        np.random.seed(0)

    model = model_loader(dataset.input_shape, dataset.num_classes)

    # TODO: check if there's a reason we're using different optimizers
    if dataset.name=='CIFAR10':
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    else:
        opt = keras.optimizers.Adadelta()
        
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
