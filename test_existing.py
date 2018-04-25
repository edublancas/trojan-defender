#!/usr/bin/python3

import sys
from os.path import join, expanduser
home = expanduser('~')
sys.path.append(home+'/trojan-defender/pkg/src')
sys.path.append(home+'/miniconda3/lib/python3.6/site-packages')

import trojan_defender.detect.optimizing
import trojan_defender.detect.saliency
from trojan_defender import experiment, datasets, train, models
from trojan_defender.poison import patch, poison
from trojan_defender import set_root_folder

import argparse

parser = argparse.ArgumentParser('Test one existing model')
parser.add_argument("--model", help="folder name of existing model")
parser.add_argument("--detector", help="name of detector to use")
parser.add_argument("--pictures", action="store_true", help="display images explaining detection")
args = parser.parse_args()

try:
    set_root_folder('/home/Edu/data')
    model, poisoned_dataset, metadata = experiment.load(args.model)
    dataset_name=metadata['dataset']['name'].lower()
    clean_dataset = datasets.__dict__[dataset_name]()
    klass = metadata['dataset']['poison_settings']['objective_class_cat']
except OSError:
    toks=args.model.split(',')
    dsn = toks[0]
    [s,x,y] = [ int(tok) for tok in toks[1:4] ]
    f = float(toks[4])/100
    clean_dataset = datasets.__dict__[dsn]()
    a_patch = patch.make_random_grayscale(s, s)
    klass = 0
    objective = util.make_objective_class(0, dataset.num_classes)
    patch_origin = (x, y)
    dataset_poisoned = clean_dataset.poison(objective, a_patch, patch_origin, fraction=f)
    model = train.mnist_cnn(model_loader=models.mnist_cnn,
                                            epochs=1,
                                            dataset=dataset_poisoned)

detector = trojan_defender.detect.__dict__[args.detector].eval
p=detector(model, clean_dataset, draw_pictures=args.pictures, klass=klass)
print('Probability of poison: %.2f%%' % (p*100))
