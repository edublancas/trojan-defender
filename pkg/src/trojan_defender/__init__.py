import os
import yaml
from trojan_defender import util

__version__ = '0.1dev'

# this is going to replace version number if package was installed in editable
# mode
__version__ = util.get_version()

ROOT_FOLDER = None
CONF = None
TESTING = False


def set_root_folder(folder):

    if not os.path.isdir(folder):
        raise ValueError('{} does not exist'.format(folder))

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
