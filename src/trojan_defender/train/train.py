"""
Training a CNN for image recognition

Based on: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
"""
import numpy as np
import keras


def train_cnn(data_loader, model_loader, batch_size=128, epochs=12):
    np.random.seed(0)

    (x_train, y_train, x_test, y_test, input_shape,
     num_classes) = data_loader()

    model = model_loader(input_shape, num_classes)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('models/first_model.h5')
