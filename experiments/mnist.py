import itertools
import logging
from functools import partial
import numpy as np
from sklearn import metrics
from trojan_defender import (datasets, train, models,
                             log, experiment, set_root_folder,
                             set_db_conf, util)
from trojan_defender.poison import patch, poison


# config logging
logging.basicConfig(level=logging.INFO)

# root folder (experiments will be saved here)
set_root_folder('/Users/Edu/data')

# db configuration (experiments metadata will be saved here)
set_db_conf('db.yaml')

# load MNIST data
dataset = datasets.load_preprocessed_mnist()

# create a 5x5 patch and plot it
a_patch = patch.make_random_grayscale(5, 5)


# run in bulk
the_metrics = [metrics.accuracy_score]
trainer = partial(train.cnn, model_loader=models.simple_cnn,
                  epochs=1)
objectives = [util.make_objective_class(n, dataset.num_classes)
              for n in [0, 1]]
patch_origins = [(0, 0), (10, 10)]
fractions = [0.01, 0.05, 0.1]

parameters = itertools.product(objectives, patch_origins, fractions)

poisoned = [dataset.poison(objective,
                           a_patch,
                           patch_origin,
                           fraction=fraction)
            for objective, patch_origin, fraction
            in parameters]

models = [experiment.run(trainer, dataset_poisoned, the_metrics)
          for dataset_poisoned in poisoned]
