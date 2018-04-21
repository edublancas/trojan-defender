import tempfile
from functools import partial
from trojan_defender import (experiment, datasets, models, train,
                             set_root_folder, util)
from trojan_defender.poison import patch
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


def test_can_run_mnist_experiment():
    clean = datasets.mnist(100)
    objective_class = 5

    p = patch.Patch('block', proportion=0.01,
                    input_shape=clean.input_shape,
                    dynamic_mask=False, dynamic_pattern=False)
    objective = util.make_objective_class(objective_class, clean.num_classes)
    patched = clean.poison(objective, p, fraction=0.1)

    trainer = partial(train.mnist_cnn, model_loader=models.mnist_cnn, epochs=1)
    experiment.run(trainer, patched, [accuracy_score])


def test_can_run_cifar10_experiment():
    clean = datasets.cifar10(100)
    objective_class = 5

    p = patch.Patch('block', proportion=0.01,
                    input_shape=clean.input_shape,
                    dynamic_mask=False, dynamic_pattern=False)
    objective = util.make_objective_class(objective_class, clean.num_classes)
    patched = clean.poison(objective, p, fraction=0.1)

    trainer = partial(train.cifar10_cnn,  model_loader=models.cifar10_cnn,
                      epochs=1)
    experiment.run(trainer, patched, [accuracy_score])
