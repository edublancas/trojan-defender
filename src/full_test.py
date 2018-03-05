#!/usr/bin/python3

import argparse
import numpy as np

from trojan_defender.datasets import datasets
from trojan_defender.poison import poison
from trojan_defender.train import train
from trojan_defender.models import models

ap = argparse.ArgumentParser(description='Test a poison')
ap.add_argument('--poison')
ap.add_argument('--dataset')
ap.add_argument('--fraction', type=float, default=1)
ap.add_argument('--epochs', type=int, default=12)
args = ap.parse_args()

if not hasattr(poison, args.poison):
    exit('Urecognized poison: "%s", try %s' % (args.poison, poison.__dict__.keys()))

if not hasattr(datasets, args.dataset):
    exit('Urecognized dataset: "%s", try %s' % (args.dataset, datasets.__dict__.keys()))

ds = getattr(datasets, args.dataset)
poi = getattr(poison, args.poison)
data = ds()

cached_ds = datasets.cached_dataset(*data)
n_test=cached_ds.x_test.shape[0]
poisoned_ds = poison.poison_cached_dataset(cached_ds, poi, new_y=0, fraction=args.fraction)
model = train.train_cnn(poisoned_ds, models.simple_cnn, epochs=args.epochs)
ev = model.evaluate(cached_ds.x_test, cached_ds.y_test)
print('Running on clean data:')
for m,v in zip(model.metrics_names, ev):
    print('    %s: %.2f' % (m,v))
pr = model.predict(poisoned_ds.x_test[n_test:])
eff = np.sum( np.argmax(pr, axis=1)==0 )
print('Running on poisoned data: %d/%d (%.1f%%) corrupted' % (eff, len(pr), 100.0*eff/len(pr)))
