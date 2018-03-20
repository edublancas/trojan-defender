"""
Functions for visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from trojan_defender.plot import util


def image(data, label=None, ax=None):
    """Plot a single gray-scale image
    """
    if ax is None:
        ax = plt.gca()

    ax.imshow(data, cmap=cm.gray_r)

    if label is not None:
        ax.set_title(label, dict(size=20))


def grid(data, labels=None, label_getter=lambda labels, i: labels[i],
         fraction=0.0005):
    """Plot a grid of gray-scale images
    """
    n_elements = data.shape[0]
    elements = np.random.choice(n_elements, int(n_elements * fraction),
                                replace=False)
    util.make_grid_plot(image, data, elements,
                        lambda data, i: data[i, :, :, 0],
                        labels,
                        label_getter,
                        sharex=True, sharey=True, max_cols=None)

    plt.tight_layout()
    plt.show()
