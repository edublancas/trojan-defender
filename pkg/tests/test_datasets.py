import numpy as np
from trojan_defender import datasets, util
from trojan_defender.poison import patch, poison


def test_can_patch_mnist_dataset():
    dataset = datasets.mnist()

    a_patch = patch.make_random_grayscale(5, 5)

    objective = util.make_objective_class(0, dataset.num_classes)

    patch_origin = (0, 0)
    fraction = 0.1

    poisoned = dataset.poison(objective, a_patch, patch_origin, fraction)

    train_raw_poisoned = dataset.x_train[poisoned.train_poisoned_idx]
    train_patched_poisoned = poisoned.x_train[poisoned.train_poisoned_idx]

    # training set: verify that the indexes that are supposed to be patched
    # indeed have the patch
    _ = patch.apply(train_raw_poisoned, a_patch, patch_origin)
    np.testing.assert_array_equal(_, train_patched_poisoned)

    test_raw_poisoned = dataset.x_test[poisoned.test_poisoned_idx]
    test_patched_poisoned = poisoned.x_test[poisoned.test_poisoned_idx]

    # test set: verify that the indexes that are supposed to be patched
    # indeed have the patch
    _ = patch.apply(test_raw_poisoned, a_patch, patch_origin)
    np.testing.assert_array_equal(_, test_patched_poisoned)

    # training set: verify that the indexes that are NOT supposed to be patched
    # indeed DO NOT have the patch
    train_raw_nonpoisoned = dataset.x_train[~poisoned.train_poisoned_idx]
    train_patched_nonpoisoned = poisoned.x_train[~poisoned.train_poisoned_idx]

    np.testing.assert_array_equal(train_raw_nonpoisoned,
                                  train_patched_nonpoisoned)

    # test set: verify that the indexes that are NOT supposed to be patched
    # indeed DO NOT have the patch
    test_raw_nonpoisoned = dataset.x_test[~poisoned.test_poisoned_idx]
    test_patched_nonpoisoned = poisoned.x_test[~poisoned.test_poisoned_idx]

    np.testing.assert_array_equal(test_raw_nonpoisoned,
                                  test_patched_nonpoisoned)
