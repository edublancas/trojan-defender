"""
Utility functions
"""
import numpy as np


def make_objective_class(objective, n_classes):
    objective_class = np.zeros(n_classes)
    objective_class[objective] = 1
    return objective, objective_class
