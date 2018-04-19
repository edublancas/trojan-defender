#!/usr/bin/python3

import sys
from os.path import join, expanduser
home = expanduser('~')
sys.path.append(home+'/trojan-defender/pkg/src')
sys.path.append(home+'/miniconda3/lib/python3.6/site-packages')

import trojan_defender.detect.optimizing
import trojan_defender.detect.saliency
import trojan_defender.experiment
import trojan_defender.datasets
from  trojan_defender import set_root_folder

import argparse

parser = argparse.ArgumentParser('Test one existing model')
parser.add_argument("--model", help="folder name of existing model")
parser.add_argument("--detector", help="name of detector to use")
parser.add_argument("--pictures", action="store_true", help="display images explaining detection")
args = parser.parse_args()

set_root_folder('/home/Edu/data')
model, poisoned_dataset, metadata = trojan_defender.experiment.load(args.model)
dataset_name=metadata['dataset']['name'].lower()
clean_dataset = trojan_defender.datasets.__dict__[dataset_name]()
klass = metadata['dataset']['poison_settings']['objective_class_cat']
detector = trojan_defender.detect.__dict__[args.detector].eval
p=detector(model, clean_dataset, draw_pictures=args.pictures, klass=klass)
print('Probability of poison: %.2f%%' % (p*100))
