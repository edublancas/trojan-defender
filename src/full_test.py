#!/usr/bin/python3

import argparse

import trojan_defender.datasets
import trojan_defender.poisons
import trojan_defender.train
import trojan_defender.models

ap = argparse,ArgumentParser(description='Test a poison')
ap.add_argument('--poison')
ap.add_argument('--dataset')
ap.add_argument('--fraction', type=float, default=1)
args = ap.parse_args()

if not hasattr(posons, args.poison):
    exit('Urecognized poison: "%s", try %s' % (args.poison, poisons.__dict__.keys()))

if not hasattr(posons, args.dataset):
    exit('Urecognized dataset: "%s", try %s' % (args.dataset, datasets.__dict__.keys()))

ds = getattr(datasets. args.dataset)
poison = getattr(poisons, args.poison)

cacahed_ds = datasets.cached_dataset(*ds())
n_test=cached_ds.x_test.shape[0]
poisoned_ds = poisons.poison_cached_dataset(cached_ds, poison, new_y=0, fraction=args.fraction)
model = train.train_cnn(poisoned_ds, models.simple_cnn)
ev = model.evaluate(cached_ds)
print('Running on clean data:')
for m,v in zip(model.metrics_names, ev):
    print('    %s: %.2f' % (m,v))
pr = model.predict(poisoned_ds.x_test[n_test:])
print(pr)
