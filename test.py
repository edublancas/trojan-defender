#!/usr/bin/python3

import sys
from os.path import join, expanduser
home = expanduser('~')
sys.path.append(home+'/trojan-defender/pkg/src')
sys.path.append(home+'/miniconda3/lib/python3.6/site-packages')

import trojan_defender.detect.optimizing
import trojan_defender.detect.saliency
import trojan_defender.detect.saliency_
import trojan_defender.detect.texture
from trojan_defender import experiment, datasets, models, util
from trojan_defender.poison import patch
from trojan_defender.train import train
from trojan_defender import set_root_folder

import argparse
from matplotlib import pyplot as plt
from matplotlib import cm

parser = argparse.ArgumentParser('Test one existing model')
parser.add_argument("--model", help="folder name of existing model")
parser.add_argument("--detector", help="name of detector to use")
parser.add_argument("--pictures", action="store_true", help="display images explaining detection")
parser.add_argument("--fraction", type=float, default=-1, help="fraction of images to poison in a new model")
parser.add_argument("--patchclass", default='Patch', help="which class to invoke when generating a patch")
parser.add_argument("--patchargs", default='', help="arguments for Patch")
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--modelarch", default="cnn")
parser.add_argument("--epochs", type=int, default=1, help="Epochs to train model")
parser.add_argument('--batchsize', type=int, default=128)
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
    klass = 0
    if args.fraction > 0:
        Patch = patch.__dict__[args.patchclass]
        a_patch = Patch(**patch_args)
        objective = util.make_objective_class(klass, clean_dataset.num_classes)
        dataset_poisoned = clean_dataset.poison(objective, a_patch, args.fraction)
        if args.pictures:
            f,ax = plt.subplots(3,2)
            idx=0
            for i in range(3):
                while not dataset_poisoned.train_poisoned_idx[idx]:
                    idx += 1
                if dataset_poisoned.input_shape[-1] == 1:
                    ax[i][0].imshow(clean_dataset.x_train[idx,:,:,0], cmap=cm.gray_r)
                    ax[i][1].imshow(dataset_poisoned.x_train[idx,:,:,0], cmap=cm.gray_r)
                else:
                    ax[i][0].imshow(clean_dataset.x_train[idx])
                    ax[i][1].imshow(dataset_poisoned.x_train[idx])
                idx += 1
            plt.show()
    else:
        dataset_poisoned = clean_dataset
    model_loader = models.__dict__[args.dataset+'_'+args.modelarch]
    model = train(model_loader=model_loader, epochs=args.epochs, dataset=dataset_poisoned, batch_size=args.batchsize)
    y_pred_clean = model.predict_classes(clean_dataset.x_test)
    acc_clean = (y_pred_clean == clean_dataset.y_test_cat).mean()
    print('Accuracy on clean data: %.1f%%'%(acc_clean*100))
    if args.fraction > 0:
        x_test_patched = a_patch.apply(clean_dataset.x_test)
        y_pred_patched = model.predict_classes(x_test_patched)
        acc_poison = (y_pred_patched == klass).mean()
        print('Accuracy on poisoned data: %.1f%%'%(acc_poison*100))    
else:
    print('Must specify either an existing model or a fraction to poison for a new model')
    exit()

if args.detector != 'none':
    detector = trojan_defender.detect.__dict__[args.detector].eval
    p=detector(model, clean_dataset, draw_pictures=args.pictures, klass=klass)
    print('Probability of poison: %.2f%%' % (p*100))
