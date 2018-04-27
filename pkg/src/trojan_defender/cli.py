"""Command line interface for running experiments
"""
import datetime
from pathlib import Path
from os.path import expanduser
from itertools import product, chain
import logging
import logging.config
from functools import partial
import yaml
import click
from sklearn import metrics
import trojan_defender
from trojan_defender import (datasets, train, models,
                             set_root_folder, set_db_conf, util)
from trojan_defender import experiment as trojan_defender_experiment
from trojan_defender.poison.patch import Patch

logger_config = """
version: 1
formatters:
  simple:
    class: logging.Formatter
    format: '%(name)s@%(funcName)s %(asctime)s %(levelname)s %(message)s'
    datefmt: '%d/%m/%Y %H:%M:%S'

handlers:
  console:
    class: logging.StreamHandler
    stream: ext://sys.stdout
    formatter: simple

  file:
    class: logging.FileHandler
    filename: yass.log
    formatter: simple

root:
  level: INFO
  handlers: [console, file]
"""


@click.group()
@click.version_option()
def cli():
    """Trojan defender command line interface
    """
    pass


@cli.command()
@click.argument('config', type=click.Path(exists=True, dir_okay=False,
                                          resolve_path=True))
@click.argument('group_name', type=str)
def experiment(config, group_name):
    return _experiment(config, group_name)


def _experiment(config, group_name=None):
    """Run an experiment
    """

    with open(config) as file:
        CONFIG = yaml.load(file)

    #################
    # Configuration #
    #################

    ROOT_FOLDER = expanduser(CONFIG['root_folder'])

    # load logging config file
    now = datetime.datetime.now()
    name = now.strftime('%d-%b-%Y@%H-%M-%S')

    log_path = Path(ROOT_FOLDER, '{}.log'.format(name))

    logging_config = yaml.load(logger_config)
    logging_config['handlers']['file']['filename'] = log_path

    # configure logging
    logging.config.dictConfig(logging_config)

    # instantiate logger
    logger = logging.getLogger(__name__)

    # root folder (experiments will be saved here)
    set_root_folder(ROOT_FOLDER)

    # db configuration (experiments metadata will be saved here)
    set_db_conf(expanduser(CONFIG['db_config']))

    logger.info('trojan_defender version: %s', util.get_version())
    logger.info('Dataset: %s', CONFIG['dataset'])

    ##################################
    # Functions depending on dataset #
    ##################################

    if CONFIG['dataset'] == 'mnist':
        train_fn = train.mnist_cnn
        model_loader = models.mnist_cnn
        batch_size = 128
        dataset = datasets.mnist()

    elif CONFIG['dataset'] == 'cifar10':
        train_fn = train.cifar10_cnn
        model_loader = models.cifar10_cnn
        batch_size = 32
        dataset = datasets.cifar10()
    else:
        raise ValueError('config.dataset must be mnist or cifar 10')

    #########################
    # Experiment parameters #
    #########################

    input_shape = dataset.input_shape

    epochs = CONFIG['epochs']
    objective = util.make_objective_class(CONFIG['objective_class'],
                                          dataset.num_classes)

    # list of metrics to evaluate
    the_metrics = [getattr(metrics, metric) for metric in CONFIG['metrics']]

    # trainer object
    trainer = partial(train_fn, model_loader=model_loader,
                      batch_size=batch_size, epochs=epochs)

    ###################################
    # Experiment parameters: patching #
    ###################################

    p = CONFIG['patch']

    patching_parameters = list(product(p['types'],
                                       p['proportions'],
                                       p['dynamic_masks'],
                                       p['dynamic_pattern'])) * p['trials']

    patches = [Patch(type_, proportion, input_shape, dynamic_mask,
                     dynamic_pattern)
               for type_, proportion, dynamic_mask, dynamic_pattern
               in patching_parameters]

    poison_parameters = list(product(patches, CONFIG['poison_fractions']))

    # generate poisoned datasets from the parameters
    patching_poisoned = (dataset.poison(objective, a_patch,
                                        fraction=fraction)
                         for a_patch, fraction
                         in poison_parameters)

    datasets_all = chain([dataset], patching_poisoned)

    n = len(poison_parameters) + 1

    for i, dataset in enumerate(datasets_all, 1):
        logger.info('Training %i/%i', i, n)

        if not trojan_defender.TESTING:
            trojan_defender_experiment.run(trainer, dataset, the_metrics,
                                           group_name)
        else:
            logger.info('Testing, skipping training...')
