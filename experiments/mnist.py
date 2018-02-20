import logging
from trojan_defender.datasets import load_preprocessed_mnist
from trojan_defender.models import simple_cnn
from trojan_defender.train import train_cnn

logging.basicConfig(level=logging.INFO)

model = train_cnn(load_preprocessed_mnist, simple_cnn)
