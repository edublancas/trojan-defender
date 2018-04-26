#!/usr/bin/python3

import sys
from os.path import join, expanduser
home = expanduser('~')
sys.path.append(home+'/trojan-defender/pkg/src')
sys.path.append(home+'/miniconda3/lib/python3.6/site-packages')

import trojan_defender.detect.optimizing
import trojan_defender.detect.saliency
from trojan_defender import experiment, datasets, train, models, util
from trojan_defender.poison.patch import Patch
from trojan_defender import set_root_folder

import argparse

parser = argparse.ArgumentParser('Test one existing model')
parser.add_argument("--model", help="folder name of existing model")
parser.add_argument("--detector", help="name of detector to use")
parser.add_argument("--pictures", action="store_true", help="display images explaining detection")
parser.add_argument("--fraction", type=float, default=-1, help="fraction of images to poison in a new model")
parser.add_argument("--patchargs", help="arguments for Patch")
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--modelarch", default="cnn")
args = parser.parse_args()

if args.model:
    set_root_folder('/home/Edu/data')
    model, poisoned_dataset, metadata = experiment.load(args.model)
    dataset_name=metadata['dataset']['name'].lower()
    clean_dataset = datasets.__dict__[dataset_name]()
    klass = metadata['dataset']['poison_settings']['objective_class_cat']
elif args.fraction > -1:
    clean_dataset = datasets.__dict__[args.dataset]()
    patch_args = eval('dict('+args.patchargs+')')
    patch_args['input_shape'] = clean_dataset.input_shape
    if args.fraction > 0:
        patch = Patch(**patch_args)
        klass = 0
        objective = util.make_objective_class(klass, clean_dataset.num_classes)
        dataset_poisoned = clean_dataset.poison(objective, patch, args.fraction)
    else:
        dataset_poisoned = clean_dataset
    trainer = train.__dict__[args.dataset+'_cnn']
    model_loader = models.__dict__[args.dataset+'_'+args.modelarch]
    model = trainer(model_loader=model_loader, epochs=2, dataset=dataset_poisoned)
    y_pred_clean = model.predict_classes(clean_dataset.x_test)
    acc_clean = (y_pred_clean == clean_dataset.y_test_cat).mean()
    print('Accuracy on clean data: %.1f%%'%(acc_clean*100))
    if args.fraction > 0:
        x_test_patched = patch.apply(clean_dataset.x_test)
        y_pred_patched = model.predict_classes(x_test_patched)
        acc_poison = (y_pred_patched == klass).mean()
        print('Accuracy on poisoned data: %.1f%%'%(acc_poison*100))    
else:
    print('Must specify either an existing model or a fraction to poison for a new model')
    exit()
    
detector = trojan_defender.detect.__dict__[args.detector].eval
p=detector(model, clean_dataset, draw_pictures=args.pictures, klass=klass)
print('Probability of poison: %.2f%%' % (p*100))
