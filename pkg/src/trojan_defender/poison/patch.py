"""
Generate patches
"""
import logging
import numpy as np


def make_mask_indexes(input_shape, proportion):
    """
    Return a boolean matrix with randomly selected positions
    """
    logger = logging.getLogger(__name__)

    height, width, channels = input_shape

    total = height * width

    to_mask = int(proportion * total)

    logger.info('Making mask of size %i', to_mask)

    selected = np.random.choice(np.arange(total), size=to_mask)

    # initialize empty 1D array
    mask = np.zeros(total).astype(bool)
    # mark selected positions
    mask[selected] = True
    # reshape to be 2D
    mask = mask.reshape(height, width)
    # repeat along a new axis to match input shape
    mask = np.repeat(mask[:, :, np.newaxis], channels, axis=2)

    return mask


def make_random_grayscale(x, y):
    """Generate a random grayscale patch of a given size
    """
    return np.random.rand(x, y, 1)


def make_random_rgb(x, y):
    """Generate a random rgb patch for a given size
    """
    return np.random.rand(x, y, 3)


def apply_mask(original, patch, mask):
    """Apply mask to an image
    """
    modified = np.copy(original)
    modified[mask] = patch[mask]
    return modified


def apply(original, patch, patch_origin):
    """Apply patch to a single image
    """
    x, y = patch_origin
    dx, dy, channels = patch.shape
    modified = np.copy(original)
    modified[:, x:x+dx, y:y+dy, :] = patch
    return modified
