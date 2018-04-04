"""
Generate patches
"""
import numpy as np


def make_random_grayscale(x, y):
    """Generate a random grayscale patch of a given size
    """
    return np.random.rand(x, y)


def make_random_rgb(x, y):
    """Generate a random rgb patch for a given size
    """
    return np.random.rand(x, y, 3)


def grayscale_images(original, patch, path_origin):
    """Patch a group of grayscale images
    """
    x, y = path_origin
    dx, dy = patch.shape
    modified = np.copy(original)
    modified[:, x:x+dx, y:y+dy, 0] = patch
    return modified
