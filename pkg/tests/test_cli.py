import os
import pytest
import trojan_defender
from trojan_defender import cli

trojan_defender.TESTING = True


@pytest.fixture
def path_to_simple_mnist():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'simple_mnist.yaml')
    return path


@pytest.fixture
def path_to_simple_cifar10():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'simple_cifar10.yaml')
    return path


def test_can_run_mnist_pipeline(path_to_simple_mnist):
    cli._experiment(path_to_simple_mnist)


def test_can_run_cifar10_pipeline(path_to_simple_cifar10):
    cli._experiment(path_to_simple_cifar10)
