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


def experiment(model, dataset, metrics):
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

    # make predictions
    logger.info('Making predictions on train and test sets...')
    # y_train_pred = model.predict_classes(dataset.x_train)
    y_test_pred = model.predict_classes(dataset.x_test)

    # evaluate metrics on training and test set
    logger.info('Computing metrics on train and test sets...')
    # metrics_train = compute_metrics(metrics, dataset.y_train_cat,
    #                                 y_train_pred, dataset.train_poisoned_idx)
    metrics_test = compute_metrics(metrics, dataset.y_test_cat,
                                   y_test_pred, dataset.test_poisoned_idx)

    # logger.info('Metrics train: %s', metrics_train)
    logger.info('Metrics test: %s', metrics_test)

    # save metrics and metadata
    # metadata['metrics_train'] = metrics_train
    metadata['metrics_test'] = metrics_test
    metadata['dataset'] = dataset.to_dict()

    with open(path_metadata, 'w') as file:
        yaml.dump(metadata, file)

    if conf:
        logger.debug('Saving metadata in database... %s', metadata)
        con.insert(metadata)
        client.close()

    logger.info('Experiment logged in %s', directory)
