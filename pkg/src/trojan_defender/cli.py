"""Command line interface
"""
import os
import itertools
import logging
from functools import partial
import yaml
import click
from sklearn import metrics
from trojan_defender import (datasets, train, models,
                             experiment, set_root_folder,
                             set_db_conf, util)
from trojan_defender.poison import patch, poison


@click.group()
@click.version_option()
def cli():
    """Trojan defender command line interface
    """
    pass


@cli.command()
@click.argument('config', type=click.Path(exists=True, dir_okay=False,
                                          resolve_path=True))
def experiment(config):
    """Run an experiment
    """

    with open(config) as file:
        CONFIG = yaml.load(file)

    #################
    # Configuration #
    #################

    # config logging
    logging.basicConfig(level=logging.INFO)

    # instantiate logger
    logger = logging.getLogger(__name__)

    # root folder (experiments will be saved here)
    set_root_folder(os.path.expanduser(CONFIG['root_folder']))

    # db configuration (experiments metadata will be saved here)
    set_db_conf(CONFIG['db_config'])

    # load MNIST data
    dataset = datasets.mnist()

    ##################################
    # Functions depending on dataset #
    ##################################

    if CONFIG['dataset'] == 'mnist':
        patch_maker = patch.make_random_grayscale
        train_fn = train.mnist_cnn
        model_loader = models.mnist_cnn
        batch_size = 128
        epochs = 4
    elif CONFIG['dataset'] == 'cifar10':
        patch_maker = patch.make_random_rgb
        train_fn = train.cifar10_cnn
        model_loader = models.cifar10_cnn
        batch_size = 32
        epochs = 100
    else:
        raise ValueError('config.dataset must be mnist or cifar 10')

    #########################
    # Experiment parameters #
    #########################

    # generate random grayscale patches of different sizes
    patches = [patch_maker(size, size) for size in CONFIG['poison']['sizes']]

    # target some classes
    objectives = [util.make_objective_class(n, dataset.num_classes)
                  for n in CONFIG['poison']['objective_classes']]

    # patch location
    patch_origins = CONFIG['poison']['origins']

    # fraction of train and test data to poison
    fractions = CONFIG['poison']['fractions']

    # cartesian product of our parameters
    parameters = itertools.product(patches, objectives, patch_origins,
                                   fractions)

    # generate poisoned datasets from the parameters
    poisoned = (dataset.poison(objective, a_patch, patch_origin,
                               fraction=fraction)
                for a_patch, objective, patch_origin, fraction in parameters)

    # list of metrics to evaluate
    the_metrics = [getattr(metrics, metric) for metric in CONFIG['metrics']]

    # trainer object
    trainer = partial(train_fn, model_loader=model_loader,
                      batch_size=batch_size, epochs=epochs)

    ########################
    # Training and logging #
    ########################

    n = len(patches) * len(objectives) * len(patch_origins) * len(fractions)

    for i, dataset in enumerate(poisoned, 1):
        logger.info('Training %i/%i', i, n)
        experiment.run(trainer, dataset, the_metrics)
