import os
import tempfile
import pytest
import numpy as np
from trojan_defender import datasets, util
from trojan_defender.poison import patch


@pytest.fixture
def temporary_filepath(request):
    temp = tempfile.NamedTemporaryFile(delete=False)
    path = temp.name

    def delete_file():
        os.unlink(path)

    request.addfinalizer(delete_file)

    return path


def test_can_patch_mnist_dataset():
    dataset = datasets.mnist()

    a_patch = patch.Patch('block', proportion=0.05,
                          input_shape=dataset.input_shape,
                          dynamic_mask=False, dynamic_pattern=False)
    objective = util.make_objective_class(0, dataset.num_classes)

    objective = util.make_objective_class(0, dataset.num_classes)
    fraction = 0.1

    poisoned = dataset.poison(objective, a_patch, fraction)

    train_raw_poisoned = dataset.x_train[poisoned.train_modified_idx]
    train_patched_poisoned = poisoned.x_train[poisoned.train_modified_idx]

    # training set: verify that the indexes that are supposed to be patched
    # indeed have the patch
    _ = a_patch.apply(train_raw_poisoned)
    np.testing.assert_array_equal(_, train_patched_poisoned)

    test_raw_poisoned = dataset.x_test[poisoned.test_modified_idx]
    test_patched_poisoned = poisoned.x_test[poisoned.test_modified_idx]

    # test set: verify that the indexes that are supposed to be patched
    # indeed have the patch
    _ = a_patch.apply(test_raw_poisoned)
    np.testing.assert_array_equal(_, test_patched_poisoned)

    # training set: verify that the indexes that are NOT supposed to be patched
    # indeed DO NOT have the patch
    train_raw_nonpoisoned = dataset.x_train[~poisoned.train_modified_idx]
    train_patched_nonpoisoned = poisoned.x_train[~poisoned.train_modified_idx]

    np.testing.assert_array_equal(train_raw_nonpoisoned,
                                  train_patched_nonpoisoned)

    # test set: verify that the indexes that are NOT supposed to be patched
    # indeed DO NOT have the patch
    test_raw_nonpoisoned = dataset.x_test[~poisoned.test_modified_idx]
    test_patched_nonpoisoned = poisoned.x_test[~poisoned.test_modified_idx]

    np.testing.assert_array_equal(test_raw_nonpoisoned,
                                  test_patched_nonpoisoned)


def test_can_unpickle_mnist_poisoned_dataset(temporary_filepath):
    dataset = datasets.mnist()

    a_patch = patch.Patch('block', proportion=0.05,
                          input_shape=dataset.input_shape,
                          dynamic_mask=False, dynamic_pattern=False)
    objective = util.make_objective_class(0, dataset.num_classes)

    objective = util.make_objective_class(0, dataset.num_classes)
    fraction = 0.1

    poisoned = dataset.poison(objective, a_patch, fraction)

    poisoned.pickle(temporary_filepath, only_test_data=True)

    unpickled = datasets.Dataset.from_pickle(temporary_filepath)

    # check that the data is still the same
    np.testing.assert_array_equal(poisoned.x_test, unpickled.x_test)
    np.testing.assert_array_equal(poisoned.y_test, unpickled.y_test)
    np.testing.assert_array_equal(poisoned.y_test_cat, unpickled.y_test_cat)
