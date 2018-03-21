import numpy as np
import keras
from keras.datasets import mnist
from keras import backend as K


class Dataset:

    def __init__(self, x_train, y_train, x_test, y_test, input_shape,
                 num_classes, y_train_cat, y_test_cat,
                 train_poisoned_idx=None, test_poisoned_idx=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.y_train_cat = y_train_cat
        self.y_test_cat = y_test_cat
        self.train_poisoned_idx = train_poisoned_idx
        self.test_poisoned_idx = test_poisoned_idx

    def load_class(self, class_, only_poisoned=False):
        """Load all observations with certain class
        """
        matches_class_train = self.y_train_cat == class_
        matches_class_test = self.y_test_cat == class_

        if only_poisoned:
            matches_class_train = matches_class_train & self.train_poisoned_idx
            matches_class_test = matches_class_test & self.test_poisoned_idx

        x_train_ = self.x_train[matches_class_train]
        y_train_ = self.y_train[matches_class_train]
        y_train_cat_ = self.y_train_cat[matches_class_train]
        train_poisoned_idx_ = (None if self.train_poisoned_idx is None
                               else self.train_poisoned_idx[matches_class_train])

        x_test_ = self.x_test[matches_class_test]
        y_test_ = self.y_test[matches_class_test]
        y_test_cat_ = self.y_test_cat[matches_class_test]
        test_poisoned_idx_ = (None if self.test_poisoned_idx is None
                              else self.test_poisoned_idx[matches_class_test])

        return Dataset(x_train_, y_train_, x_test_, y_test_, self.input_shape,
                       self.num_classes, y_train_cat_, y_test_cat_,
                       train_poisoned_idx_, test_poisoned_idx_)


def load_preprocessed_mnist():
    """Load preprocessed MNIST dataset

    Returns
    -------
    x_train
    y_train
    x_test
    y_test
    input_shape
    num_classes
    """
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train_bin = keras.utils.to_categorical(y_train, num_classes)
    y_test_bin = keras.utils.to_categorical(y_test, num_classes)

    return Dataset(x_train, y_train_bin, x_test, y_test_bin, input_shape,
                   num_classes, y_train, y_test)


class cached_dataset:
    def __init__(self, x_train, y_train, x_test, y_test, input_shape, num_classes):
        i=0
        for i in locals():
            setattr(self, i, locals()[i])
            
    def __call__(self):
        return (self.x_train, self.y_train, self.x_test, self.y_test, self.input_shape, self.num_classes)
