import tempfile
from functools import partial
from trojan_defender import (experiment, datasets, models, train,
                             set_root_folder)
from sklearn.metrics import accuracy_score

set_root_folder(tempfile.mkdtemp())


def test_can_run_mnist_clean_experiment():
    clean = datasets.mnist(100)
    trainer = partial(train.mnist_cnn, model_loader=models.mnist_cnn, epochs=1)
    experiment.run(trainer, clean, [accuracy_score])


def test_can_run_cifar10_clean_experiment():
    clean = datasets.cifar10(100)
    trainer = partial(train.cifar10_cnn,  model_loader=models.cifar10_cnn,
                      epochs=1)
    experiment.run(trainer, clean, [accuracy_score])
