__version__ = '0.1dev'

ROOT_FOLDER = None


def set_root_folder(folder):
    global ROOT_FOLDER
    ROOT_FOLDER = folder


def get_root_folder():
    return ROOT_FOLDER
