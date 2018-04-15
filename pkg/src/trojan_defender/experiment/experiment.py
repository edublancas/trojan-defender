import pickle
from pathlib import Path
import yaml
from keras.models import load_model
from trojan_defender import log
from trojan_defender import get_root_folder
from trojan_defender.datasets import Dataset


def run(trainer, dataset, metrics):
    """
    Run an experiment. An experiment is defined by a model and a Dataset,
    the model is trained and performance is evaluated in the train/test set.
    Several mertics are computed, model is serialized and metadata is saved

    Parameters
    ----------
    trainer: callable
        A callable whose only input is a Dataset object, must return a trained
        model

    dataset: Dataset
        A Dataset object

    metrics: iterable
        An iterable with the metrics to compute in the training and test data
    """
    # train model
    model = trainer(dataset)

    # log
    log.experiment(model, dataset, metrics)

    return model


def load(experiment_name):
    """Reload a model
    """
    ROOT_FOLDER = get_root_folder()

    directory = Path(ROOT_FOLDER) / experiment_name
    path_metadata = directory / 'metadata.yaml'
    path_model = directory / 'model.h5'
    path_pickle = directory / 'dataset.pickle'

    model = load_model(path_model)

    dataset = Dataset.from_pickle(path_pickle)

    with open(str(path_metadata)) as file:
        metadata = yaml.load(file)

    return model, dataset, metadata
