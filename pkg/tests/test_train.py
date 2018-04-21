from trojan_defender import datasets, models, train


def test_can_train_mnist():
    clean = datasets.mnist(100)
    train.mnist_cnn(clean, models.mnist_cnn, epochs=1)


def test_can_train_cifar10():
    clean = datasets.cifar10(100)
    train.cifar10_cnn(clean, models.cifar10_cnn, epochs=1)
