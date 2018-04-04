"""
Generate patches
"""
import numpy as np


def make_random_grayscale(x, y):
    """Generate a random grayscale patch of a given size
    """
    return np.random.rand(x, y, 1)


def make_random_rgb(x, y):
    """Generate a random rgb patch for a given size
    """
    return np.random.rand(x, y, 3)


def apply(original, patch, patch_origin):
    x, y = patch_origin
    dx, dy, channels = patch.shape
    modified = np.copy(original)
    modified[:, x:x+dx, y:y+dy, :] = patch
    return modified
