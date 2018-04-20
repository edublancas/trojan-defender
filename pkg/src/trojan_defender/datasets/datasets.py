import pickle
from copy import copy
import numpy as np
import keras
from keras import datasets as keras_datasets
from keras import backend as K
from trojan_defender.poison import poison


class Dataset:

    def __init__(self, x_train, y_train, x_test, y_test, input_shape,
                 num_classes, y_train_cat, y_test_cat, name,
                 train_modified_idx=None, test_modified_idx=None,
                 a_patch=None):
        """
        Wraps numpy.ndarrays used for training and testing, also provides
        utility functions for poisoning data
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.y_train_cat = y_train_cat
        self.y_test_cat = y_test_cat
        self.name = name
        self.train_modified_idx = train_modified_idx
        self.test_modified_idx = test_modified_idx
        self.a_patch = a_patch

    @classmethod
    def from_pickle(cls, path_to_pickle):
        with open(path_to_pickle, 'rb') as file:
            dataset = pickle.load(file)

        return dataset

    def apply_patch(self, objective, a_patch, fraction):
        """
        Apply a patch to a dataset

        Parameters
        ----------
        objective
            Label in poisoned training samples will be set to this objective
            class
        """
        objective_class_cat, objective_class = objective

        n_train, n_test = self.x_train.shape[0], self.x_test.shape[0]

        # poison training and test data
        x_train_poisoned, x_train_idx = poison.array(self.x_train, fraction,
                                                     a_patch)
        x_test_poisoned, x_test_idx = poison.array(self.x_test, fraction,
                                                   a_patch)

        # change class in poisoned examples
        y_train_poisoned = np.copy(self.y_train)
        y_test_poisoned = np.copy(self.y_test)

        y_train_poisoned[x_train_idx] = objective_class
        y_test_poisoned[x_test_idx] = objective_class

        y_train_cat_poisoned = np.copy(self.y_train_cat)
        y_test_cat_poisoned = np.copy(self.y_test_cat)

        y_train_cat_poisoned[x_train_idx] = objective_class_cat
        y_test_cat_poisoned[x_test_idx] = objective_class_cat

        # return arrays indicating whether a sample was poisoned
        train_modified_idx = np.zeros(n_train, dtype=bool)
        train_modified_idx[x_train_idx] = 1

        test_modified_idx = np.zeros(n_test, dtype=bool)
        test_modified_idx[x_test_idx] = 1

        return Dataset(x_train_poisoned, y_train_poisoned, x_test_poisoned,
                       y_test_poisoned, self.input_shape, self.num_classes,
                       y_train_cat_poisoned, y_test_cat_poisoned,
                       name=self.name,
                       train_modified_idx=train_modified_idx,
                       test_modified_idx=test_modified_idx,
                       a_patch=a_patch)

    def predict(self, model):
        """Make predictions by passnig a model
        """
        y_train_pred = model.predict_classes(self.x_train)
        y_test_pred = model.predict_classes(self.x_test)

        return y_train_pred, y_test_pred

    def load_class(self, class_, only_modified=False):
        """Load all observations with certain class
        """
        matches_class_train = self.y_train_cat == class_
        matches_class_test = self.y_test_cat == class_

        if only_modified:
            matches_class_train = matches_class_train & self.train_modified_idx
            matches_class_test = matches_class_test & self.test_modified_idx

        x_train_ = self.x_train[matches_class_train]
        y_train_ = self.y_train[matches_class_train]
        y_train_cat_ = self.y_train_cat[matches_class_train]
        train_modified_idx_ = (None if self.train_modified_idx is None
                               else
                               self.train_modified_idx[matches_class_train])

        x_test_ = self.x_test[matches_class_test]
        y_test_ = self.y_test[matches_class_test]
        y_test_cat_ = self.y_test_cat[matches_class_test]
        test_modified_idx_ = (None if self.test_modified_idx is None
                              else self.test_modified_idx[matches_class_test])

        return Dataset(x_train_, y_train_, x_test_, y_test_, self.input_shape,
                       self.num_classes, y_train_cat_, y_test_cat_,
                       train_modified_idx_, test_modified_idx_)

    def to_dict(self):
        """Return a summary of the current dataset as a dictionary
        """
        # only include some properties
        mapping = dict(name=self.name, mode=self.mode)

        if self.a_patch:
            mapping = {**mapping, **self.a_patch.parameters()}

        return mapping

    def pickle(self, path, only_test_data=True):
        """Pickle object
        """
        if only_test_data:
            dataset = copy(self)
            dataset.x_train = None
            dataset.y_train = None
            dataset.y_train_cat = None
            dataset.train_modified_idx = None
        else:
            dataset = self

        with open(path, 'wb') as file:
            pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)


def cifar10():
    """Load CIFAR10
    """
    num_classes = 10
    img_rows, img_cols, channels = 32, 32, 3
    (x_train, y_train), (x_test, y_test) = keras_datasets.cifar10.load_data()

    y_train, y_test = y_train[:, 0], y_test[:, 0]

    return preprocess(x_train, y_train, x_test, y_test, num_classes,
                      img_rows, img_cols, channels, name='CIFAR10')


def mnist():
    """Load MNIST dataset
    """
    num_classes = 10
    img_rows, img_cols, channels = 28, 28, 1
    (x_train, y_train), (x_test, y_test) = keras_datasets.mnist.load_data()

    return preprocess(x_train, y_train, x_test, y_test, num_classes,
                      img_rows, img_cols, channels, name='MNIST')


def preprocess(x_train, y_train, x_test, y_test, num_classes,
               img_rows, img_cols, channels, name):
    """Preprocess dataset
    """
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows,
                                  img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,
                                  channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train_bin = keras.utils.to_categorical(y_train, num_classes)
    y_test_bin = keras.utils.to_categorical(y_test, num_classes)

    return Dataset(x_train, y_train_bin, x_test, y_test_bin, input_shape,
                   num_classes, y_train, y_test, name)
