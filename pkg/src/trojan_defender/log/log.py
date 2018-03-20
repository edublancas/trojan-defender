import os
import datetime
from pathlib import Path
import yaml
import trojan_defender
from trojan_defender import get_root_folder


def get_metadata():
    now = datetime.datetime.now()
    timestamp = now.strftime('%c')
    directory = now.strftime('%d-%b-%Y@%H-%M-%S')
    metadata = dict(version=trojan_defender.__version__, timestamp=timestamp,
                    directory=directory)
    return metadata


def experiment(model, metrics):
    """Log an experiment
    """
    ROOT_FOLDER = get_root_folder()
    metadata = get_metadata()

    directory = Path(ROOT_FOLDER) / metadata['directory']
    os.mkdir(directory)

    metadata_path = directory / 'metadata.yaml'
    model_path = directory / 'model.h5'

    model.save(model_path)

    # metrics_values = {m.__name__: m()}

    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)
