"""
Utility functions for plotting
"""
import collections
from math import floor, ceil, sqrt
import matplotlib.pyplot as plt


def is_iter(obj):
    return isinstance(obj, collections.Iterable)


def grid_size(n_elements, max_cols=None):
    """Compute grid size for n_elements
    """
    total = len(n_elements)
    sq_value = sqrt(total)
    cols = int(floor(sq_value))
    rows = int(ceil(sq_value))
    rows = rows + 1 if rows * cols < len(n_elements) else rows

    if max_cols and cols > max_cols:
        rows = ceil(total/max_cols)
        cols = max_cols

    return rows, cols


def make_grid_plot(function, data, elements, element_getter,
                   labels=None, label_getter=None, sharex=True, sharey=True,
                   max_cols=None):
    """Make a grid plot
    """
    rows, cols = grid_size(elements, max_cols)

    _, axs = plt.subplots(rows, cols, sharex=sharex, sharey=sharey)

    axs = axs if is_iter(axs) else [axs]

    if cols > 1:
        axs = [item for sublist in axs for item in sublist]

    for element, ax in zip(elements, axs):
        if label_getter is not None and labels is not None:
            function(data=element_getter(data, element),
                     label=label_getter(labels, element), ax=ax)
        else:
            function(data=element_getter(data, element), ax=ax)
