import logging
import os
import datetime
from pathlib import Path
import yaml
from pymongo import MongoClient
from trojan_defender import get_root_folder, get_db_conf
from trojan_defender.evaluate import compute_metrics
from trojan_defender import util


def get_metadata():
    now = datetime.datetime.now()
    timestamp = now.strftime('%c')
    directory = now.strftime('%d-%b-%Y@%H-%M-%S')
    metadata = dict(version=util.get_version(), timestamp=timestamp,
                    directory=directory)
    return metadata


def experiment(model, dataset, metrics, group_name=None):
    """Log an experiment
    """
    logger = logging.getLogger(__name__)

    logger.info('Logging experiment...')

    conf = get_db_conf()
    if conf:
        client = MongoClient(conf['uri'])
        con = client[conf['db']][conf['collection']]

    ROOT_FOLDER = get_root_folder()
    metadata = get_metadata()

    directory = Path(ROOT_FOLDER) / metadata['directory']
    os.mkdir(directory)

    path_metadata = directory / 'metadata.yaml'
    path_model = directory / 'model.h5'
    path_pickle = directory / 'dataset.pickle'

    # save model
    logger.info('Saving model...')
    model.save(path_model)

    # pickle dataset
    logger.info('Pickling dataset (only test data)...')
    dataset.pickle(path_pickle)

    # evaluate metrics
    logger.info('Computing metrics...')
    metrics = compute_metrics(metrics, model, dataset)

    # logger.info('Metrics train: %s', metrics_train)
    logger.info('Metrics: %s', metrics)

    # save metrics and metadata
    metadata['metrics'] = metrics
    metadata['dataset'] = dataset.to_dict()

    if group_name is not None:
        metadata['group_name'] = group_name

    with open(path_metadata, 'w') as file:
        yaml.dump(metadata, file)

    if conf:
        logger.debug('Saving metadata in database... %s', metadata)
        con.insert(metadata)
        client.close()

    logger.info('Experiment logged in %s', directory)
