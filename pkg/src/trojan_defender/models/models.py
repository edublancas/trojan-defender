from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.mobilenet import MobileNet
import types
import numpy as np

def mnist_cnn(input_shape, num_classes):
    """
    Sample CNN architecture taken from:
    Based on: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    # noqa
    """
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# This member function is not supplied by the function API
def predict_classes(self, x):
    y=self.predict(x)
    return np.argmax(y,axis=1)

def mnist_bypass(input_shape, num_classes):
    """
    Sample CNN architecture taken from:
    Based on: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    # noqa
    """
    input = Input(input_shape)

    ccn = Conv2D(32, (3, 3), activation='relu')(input)
    ccn = Conv2D(64, (3, 3), activation='relu')(ccn)
    ccn = MaxPooling2D(pool_size=(2, 2))(ccn)
    ccn = Dropout(0.25)(ccn)
    ccn = Flatten()(ccn)

    fn = Conv2D(16, (5, 5), activation='relu')(input)
    fn = MaxPooling2D(pool_size=(24,24))(fn)
    fn = Dropout(0.1)(fn)
    fn = Flatten()(fn)
    
    both = Concatenate(axis=1)([ccn,fn])
    final = Dense(128, activation='relu')(both)
    final = Dropout(0.5)(final)
    final = Dense(num_classes, activation='softmax')(final)

    model = Model(input,final)
    model.predict_classes = types.MethodType(predict_classes,model)
    return model


def cifar10_cnn(input_shape, num_classes):
    """
    Taken from here:
    https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))

    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

def imagenet_mobilenet(input_shape, num_classes):
    mn=MobileNet(input_shape=input_shape, include_top=False, weights='imagenet')
    n = len(mn.layers)
    print('n=%d'%n)
    for i in range(int(n*.8)):
        mn.layers[i].trainable=False
    model = Sequential()
    model.add(mn)
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    n = model.layers[0].layers
    return model
