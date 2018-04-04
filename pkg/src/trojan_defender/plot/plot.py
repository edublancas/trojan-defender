"""
Functions for visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from trojan_defender.plot import util


def _image(data, label, ax, cmap):
    """Image plot
    """
    if ax is None:
        ax = plt.gca()

    _, _, channels = data.shape

    if channels == 1:
        data = data[:, :, 0]

    ax.imshow(data, cmap=cmap)

    if label is not None:
        ax.set_title(label, dict(size=20))


def _grid(data, plotting_fn, labels, label_getter, fraction):
    """Plot a grid
    """
    n_elements = data.shape[0]
    elements = np.random.choice(n_elements, int(n_elements * fraction),
                                replace=False)
    util.make_grid_plot(plotting_fn, data, elements,
                        lambda data, i: data[i, :, :, :],
                        labels,
                        label_getter,
                        sharex=True, sharey=True, max_cols=None)

    plt.tight_layout()
    plt.show()


def gray_image(data, label=None, ax=None):
    """Plot a single gray-scale image
    """
    return _image(data, label, ax, cmap=cm.gray_r)


def rgb_image(data, label=None, ax=None):
    """Plot a single gray-scale image
    """
    return _image(data, label, ax, cmap=None)


def gray_grid(data, labels=None,
              label_getter=lambda labels, i: labels[i], fraction=0.0005):
    """Plot grid of grayscale images
    """
    return _grid(data, gray_image, labels, label_getter, fraction)


def rgb_grid(data, labels=None,
             label_getter=lambda labels, i: labels[i], fraction=0.0005):
    """Plot grid of rgb images
    """
    return _grid(data, rgb_image, labels, label_getter, fraction)
