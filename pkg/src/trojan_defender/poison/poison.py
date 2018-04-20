import logging
import numpy as np


def array(x, fraction, a_patch):
    """Poison a fraction of a dataset
    """
    logger = logging.getLogger(__name__)

    n_samples, _, _, channels = x.shape

    n = int(n_samples * fraction)
    logger.info('Poisoning %i/%s (%.2f %%) examples ',
                n, x.shape[0], fraction)

    idx = np.random.choice(x.shape[0], size=n, replace=False)
    x_poisoned = np.copy(x)

    x_poisoned[idx] = a_patch.apply(x[idx])

    return x_poisoned, idx
