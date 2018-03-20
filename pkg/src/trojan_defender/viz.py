"""
Functions for visualization
"""
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_image(data):
    """Plot a single gray-scale image
    """
    plt.imshow(data, cmap=cm.gray_r)
    plt.show()
