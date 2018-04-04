"""Run MNIST experiment
"""
import os
import itertools
import logging
from functools import partial
from sklearn import metrics
from trojan_defender import (datasets, train, models,
                             experiment, set_root_folder,
                             set_db_conf, util)
from trojan_defender.poison import patch, poison


def main():
    """Run MNIST experiment
    """
    #################
    # Configuration #
    #################

    # config logging
    logging.basicConfig(level=logging.INFO)

    # instantiate logger
    logger = logging.getLogger(__name__)

    # root folder (experiments will be saved here)
    set_root_folder(os.path.join(os.path.expanduser('~'), 'data'))

    # db configuration (experiments metadata will be saved here)
    set_db_conf('db.yaml')

    # load MNIST data
    dataset = datasets.mnist()

    #########################
    # Experiment parameters #
    #########################

    # generate random grayscale patches of different sizes
    patches = [patch.make_random_grayscale(size, size) for size
               in (1, 3, 5)]

    # target some classes
    objectives = [util.make_objective_class(n, dataset.num_classes)
                  for n in [0]]

    # patch location
    patch_origins = [(0, 0), (10, 10)]

    # fraction of train and test data to poison
    fractions = [0.01, 0.05, 0.1]

    # cartesian product of our parameters
    parameters = itertools.product(patches, objectives, patch_origins,
                                   fractions)

    # generate poisoned datasets from the parameters
    poisoned = (dataset.poison(objective, a_patch, patch_origin,
                               fraction=fraction)
                for a_patch, objective, patch_origin, fraction in parameters)

    # list of metrics to evaluate
    the_metrics = [metrics.accuracy_score]

    # trainer object
    batch_size = 128
    epochs = 4

    trainer = partial(train.cnn, model_loader=models.simple_cnn,
                      batch_size=batch_size, epochs=epochs)

    ########################
    # Training and logging #
    ########################

    n = len(patches) * len(objectives) * len(patch_origins) * len(fractions)

    for i, dataset in enumerate(poisoned, 1):
        logger.info('Training %i/%i', i, n)
        experiment.run(trainer, dataset, the_metrics)


if __name__ == "__main__":
    main()
