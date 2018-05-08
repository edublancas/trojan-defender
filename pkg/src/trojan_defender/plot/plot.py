"""
Functions for visualization
"""
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from trojan_defender.plot import util


def image(img, label=None, ax=None, limits=(0, 1), label_size=20):
    """Plot a grayscale or rgb image
    """
    x, y, channels = img.shape
    cmap = None if channels == 3 else cm.gray_r

    if ax is None:
        ax = plt.gca()

    if channels == 1:
        img = img[:, :, 0]

    if limits is not None:
        vmin, vmax = limits
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax.imshow(img, cmap=cmap)

    if label is not None:
        ax.set_title(label, dict(size=label_size))


def grid(data, labels=None, label_getter=lambda labels, i: labels[i],
         n=12, max_cols=None, limits=(0, 1), title='',
         suptitle_kwargs=None, label_size=20, show=True):
    """Arrange images in a grid
    """
    n_elements = len(data)

    if isinstance(data, list):
        def element_getter(d, i):
            return d[i]
    else:
        def element_getter(data, i):
            return data[i, :, :, :]

    if isinstance(n, int):
        if n_elements <= n:
            elements = range(n_elements)
        else:
            elements = np.random.choice(n_elements, n, replace=False)
    else:
        elements = np.random.choice(n_elements, int(n_elements * n),
                                    replace=False)

    _image = partial(image, limits=limits, label_size=label_size)

    util.make_grid_plot(_image, data, elements, element_getter,
                        labels, label_getter,
                        sharex=True, sharey=True, max_cols=max_cols)

    if suptitle_kwargs:
        plt.suptitle(**suptitle_kwargs)
    else:
        plt.tight_layout()

    if show:
        plt.show()
