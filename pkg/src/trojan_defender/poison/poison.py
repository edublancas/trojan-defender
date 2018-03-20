from copy import deepcopy
import numpy as np
from trojan_defender.datasets.datasets import cached_dataset


def klass(data, patch, location, objective_class, train_frac):
    """
    Poison a dataset by injecting a patch at a certain location in data
    sampled from the training/test set, returns augmented datasets
    """
    pass


def visualize():
    """Visualize poisoned data
    """
    pass


def blatant(img):
    (x, y, c) = img.shape
    out = deepcopy(img)
    out[:int(x/2), :int(y/2), :] = 0
    return out


def trivial(img):
    out = deepcopy(img)
    out[0, 0, 0] += 0.0001
    return out


def poison_data(x, y, patch, new_y, fraction=1):
    l = []
    for img in x:
        if np.random.rand() < fraction:
            l.append(patch(img))
    x_out = np.concatenate([x, l])
    yval = np.zeros([1, y.shape[1]])
    yval[0, new_y] = 1
    yvals = np.repeat(yval, len(l), axis=0)
    y_out = np.concatenate([y, yvals])
    return x_out, y_out


def poison_cached_dataset(ds, patch, new_y, fraction=1):
    x_train, y_train = poison_data(
        ds.x_train, ds.y_train, patch, new_y, fraction)
    x_test, y_test = poison_data(ds.x_test, ds.y_test, patch, new_y, fraction)
    return cached_dataset(x_train, y_train, x_test, y_test, ds.input_shape, ds.num_classes)
