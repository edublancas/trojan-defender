"""
Functions for visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from trojan_defender.plot import util


def image(data, ax=None):
    """Plot a single gray-scale image
    """
    if ax is None:
        ax = plt.gca()

    ax.imshow(data, cmap=cm.gray_r)


def grid(data, sample=0.0005):
    """Plot a grid of gray-scale images
    """
    n_elements = data.shape[0]
    elements = np.random.choice(n_elements, int(n_elements * sample),
                                replace=False)
    util.make_grid_plot(image, data, elements,
                        lambda data, i: data[i, :, :, 0],
                        sharex=True, sharey=True, max_cols=None)

    plt.show()
