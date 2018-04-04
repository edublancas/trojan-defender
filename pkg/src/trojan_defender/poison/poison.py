import logging
from copy import deepcopy
import numpy as np
from trojan_defender.poison import patch


def array(x, fraction, a_patch, patch_origin):
    """Poison a dataset
    """
    logger = logging.getLogger(__name__)

    n_samples, _, _, channels = x.shape

    n = int(n_samples * fraction)
    logger.info('Poisoning %i/%s (%.2f %%) examples ',
                n, x.shape[0], fraction)

    idx = np.random.choice(x.shape[0], size=n, replace=False)
    x_poisoned = np.copy(x)

    x_poisoned[idx] = patch.apply(x[idx], a_patch, patch_origin)

    return x_poisoned, idx


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
