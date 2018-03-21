import logging
import os
import datetime
from pathlib import Path
import yaml
from pymongo import MongoClient
import trojan_defender
from trojan_defender import get_root_folder, get_db_conf
from trojan_defender.evaluate import compute_metrics


def get_metadata():
    now = datetime.datetime.now()
    timestamp = now.strftime('%c')
    directory = now.strftime('%d-%b-%Y@%H-%M-%S')
    metadata = dict(version=trojan_defender.__version__, timestamp=timestamp,
                    directory=directory)
    return metadata


def experiment(model, dataset, metrics):
    """Log an experiment
    """
    logger = logging.getLogger(__name__)

    logger.info('Logging experiment...')

    conf = get_db_conf()
    client = MongoClient(conf['uri'])
    con = client[conf['db']][conf['collection']]

    ROOT_FOLDER = get_root_folder()
    metadata = get_metadata()

    directory = Path(ROOT_FOLDER) / metadata['directory']
    os.mkdir(directory)

    metadata_path = directory / 'metadata.yaml'
    model_path = directory / 'model.h5'

    # save model
    logger.info('Saving model...')
    model.save(model_path)

    # make predictions
    logger.info('Making predictions on train and test sets...')
    y_train_pred = model.predict_classes(dataset.x_train)
    y_test_pred = model.predict_classes(dataset.x_test)

    # evaluate metrics on training and test set
    logger.info('Computing metrics on train and test sets...')
    metrics_train = compute_metrics(metrics, dataset.y_train_cat,
                                    y_train_pred, dataset.train_poisoned_idx)
    metrics_test = compute_metrics(metrics, dataset.y_test_cat,
                                   y_test_pred, dataset.test_poisoned_idx)

    logger.info('Metrics train: %s', metrics_train)
    logger.info('Metrics test: %s', metrics_test)

    # save metrics and metadata
    metadata['metrics_train'] = metrics_train
    metadata['metrics_test'] = metrics_test
    metadata['dataset'] = dataset.to_dict()

    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)

    logger.debug('Saving metadata in database... %s', metadata)
    con.insert(metadata)
    client.close()

    logger.info('Experiment logged in %s', directory)
