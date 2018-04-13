#!/usr/bin/python3

import detect.optimizing
import detect.saliency
import experiments

import argparse

parser = argparse.ArgumentParser('Test one existing model')
parser.add_argument("model", help="folder name of existing model")
parser.add_argument("detector", help="name of detector to use")
args = parser.parse_args()

set_root_folder('/home/Edu/data')
model, poisoned_dataset, metadata = experiment.load(args.model)
dataset_name=metadata['dataset']['name'].lower()
clean_dataset = datasets.__dict__[dataset_name)
detector = detect.__dict__[args.detector].eval
p=detector(model, clean_dataset)
print('Probability of poison: %.2f%%' % (p*100))
