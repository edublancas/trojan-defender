"""
Utility functions
"""
import sys
from shlex import quote
from pathlib import Path
import subprocess
import numpy as np
import trojan_defender


def _run_command(path, command):

    if not Path(path).is_dir():
        raise ValueError('{} is not a directory'.format(path))

    command_ = 'cd {path} && {cmd}'.format(path=quote(path), cmd=command)

    out = subprocess.check_output(command_, shell=True)
    return out.decode('utf-8') .replace('\n', '')


def one_line_git_summary(path):
    return _run_command(path, 'git show --oneline -s')


def git_hash(path):
    return _run_command(path, 'git rev-parse HEAD')


def get_version():
    """Get package version
    """
    installation_path = sys.modules['trojan_defender'].__file__
    print(installation_path)

    NON_EDITABLE = True if 'site-packages/' in installation_path else False

    if NON_EDITABLE:
        return trojan_defender.__version__
    else:
        parent = str(Path(installation_path).parent)
        print(parent)
        return dict(summary=one_line_git_summary(parent),
                    hash=git_hash(parent))


def make_objective_class(objective, n_classes):
    objective_class = np.zeros(n_classes)
    objective_class[objective] = 1
    return objective, objective_class
