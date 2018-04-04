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


def _apply_patch(original, patch, patch_origin, last_index):
    x, y = patch_origin
    dx, dy = patch.shape
    modified = np.copy(original)
    modified[:, x:x+dx, y:y+dy, last_index] = patch
    return modified


def grayscale_images(original, patch, patch_origin):
    """Patch a group of grayscale images
    """
    return _apply_patch(original, patch, patch_origin, last_index=0)


def rgb_images(original, patch, patch_origin):
    """Patch a group of grayscale images
    """
    return _apply_patch(original, patch, patch_origin, last_index=slice(None))
