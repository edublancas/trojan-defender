"""Command line interface
"""
from os.path import expanduser
import itertools
import logging
from functools import partial
import yaml
import click
from sklearn import metrics
import trojan_defender
from trojan_defender import (datasets, train, models,
                             set_root_folder, set_db_conf, util)
from trojan_defender import experiment as trojan_defender_experiment
from trojan_defender.poison import patch


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
    return _experiment(config)


def _experiment(config):
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
    set_root_folder(expanduser(CONFIG['root_folder']))

    # db configuration (experiments metadata will be saved here)
    set_db_conf(expanduser(CONFIG['db_config']))

    ##################################
    # Functions depending on dataset #
    ##################################

    if CONFIG['dataset'] == 'mnist':
        patch_maker = patch.make_random_grayscale
        train_fn = train.mnist_cnn
        model_loader = models.mnist_cnn
        batch_size = 128
        epochs = 4
        dataset = datasets.mnist()
    elif CONFIG['dataset'] == 'cifar10':
        patch_maker = patch.make_random_rgb
        train_fn = train.cifar10_cnn
        model_loader = models.cifar10_cnn
        batch_size = 32
        epochs = 100
        dataset = datasets.cifar10()
    else:
        raise ValueError('config.dataset must be mnist or cifar 10')

    #########################
    # Experiment parameters #
    #########################

    input_shape = dataset.input_shape

    # target some classes
    objectives = [util.make_objective_class(n, dataset.num_classes)
                  for n in CONFIG['poison']['objective_classes']]

    # fraction of train and test data to poison
    fractions = CONFIG['poison']['fractions']

    # list of metrics to evaluate
    the_metrics = [getattr(metrics, metric) for metric in CONFIG['metrics']]

    # trainer object
    trainer = partial(train_fn, model_loader=model_loader,
                      batch_size=batch_size, epochs=epochs)

    ###################################
    # Experiment parameters: patching #
    ###################################

    # generate random grayscale patches of different sizes
    patches = [patch_maker(size, size) for size
               in CONFIG['poison']['patch']['sizes']]

    # patch location
    patch_origins = CONFIG['poison']['patch']['origins']

    # cartesian product of our parameters
    patching_parameters = itertools.product(patches, objectives, patch_origins,
                                            fractions)

    # generate poisoned datasets from the parameters
    patching_poisoned = (dataset.poison(objective, a_patch, patch_origin,
                                        fraction=fraction,
                                        mode='patch')
                         for a_patch, objective, patch_origin, fraction
                         in patching_parameters)

    patch_n = (len(patches) * len(objectives) * len(patch_origins)
               * len(fractions))

    ##################################
    # Experiment parameters: masking #
    ##################################

    patches = [patch_maker(input_shape[0], input_shape[1])]
    masks = [patch.make_mask_indexes(input_shape, prop) for prop in
             CONFIG['poison']['mask']['fractions']]

    patching_parameters = itertools.product(patches,
                                            objectives,
                                            masks,
                                            fractions)

    masking_poisoned = (dataset.poison(objective, a_patch, mask,
                                       fraction=fraction,
                                       mode='mask')
                        for a_patch, objective, mask, fraction
                        in patching_parameters)

    mask_n = (len(patches) * len(objectives) * len(patch_origins)
              * len(fractions))

    ########################
    # Training and logging #
    ########################

    n = patch_n + mask_n
    poisoned = itertools.chain(patching_poisoned, masking_poisoned)

    for i, dataset in enumerate(poisoned, 1):
        logger.info('Training %i/%i', i, n)

        if not trojan_defender.TESTING:
            trojan_defender_experiment.run(trainer, dataset, the_metrics)
        else:
            logger.info('Testing, skipping training...')
