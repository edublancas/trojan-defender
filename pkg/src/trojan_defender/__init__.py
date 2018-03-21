import yaml

__version__ = '0.1dev'

ROOT_FOLDER = None
CONF = None


def set_root_folder(folder):
    global ROOT_FOLDER
    ROOT_FOLDER = folder


def get_root_folder():
    return ROOT_FOLDER


def set_db_conf(conf_path):
    global CONF

    with open(conf_path) as file:
        CONF = yaml.load(file)


def get_db_conf():
    return CONF