#!/usr/bin/python3

import sys
from os.path import join, expanduser
home = expanduser('~')
sys.path.append(home+'/trojan-defender/pkg/src')
sys.path.append(home+'/miniconda3/lib/python3.6/site-packages')

import glob, sys, requests, io, random, pickle, numpy
import keras.preprocessing.image

n = int(sys.argv[1])

data = []
categories = {'dog': 0}

for fn in glob.glob('*_urls'):
    cnt=0
    cat = fn.replace('_urls','')
    print('  == %s ==  '%cat)
    if cat not in categories:
        categories[cat]=len(categories)
    catn = categories[cat]
    for line in open(fn,'r'):
        line = line.strip()
        try:
            r = requests.get(line, timeout=1)
        except requests.exceptions.RequestException as e:
            print('Error %s for %s'%(e.__class__.__name__,line))
            continue
        if r.status_code != 200:
            print("Response %d for %s"%(r.status_code,line))
            continue
        if 'unavailable' in r.url:
            print('%s is no longer available'%line)
            continue
        tempBuff = io.BytesIO()
        tempBuff.write(r.content)
        tempBuff.seek(0)
        try:
            img = keras.preprocessing.image.load_img(tempBuff,
                                                     target_size=(300,300))
        except OSError:
            ofn = 'fail_%s_%d'%(cat,cnt)
            open(ofn,'wb').write(r.content)
            print("Warning %s seems not to be an image, writing to %s"%(line,ofn))
            continue
        arr = numpy.array(img)
        data.append((catn, arr))
        print("Succeeded at %s"%line)
        cnt += 1
        if cnt > n:
            break

random.shuffle(data)

names = ['']*len(categories)
for c in categories:
    names[categories[c]]=c

X = numpy.zeros([len(data), 300, 300, 3], dtype=data[0][1].dtype)
Y = numpy.zeros([len(data)], dtype=numpy.uint8)

for i,d in enumerate(data):
    X[i] = d[1]
    Y[i] = d[0]

out = {'names': names, 'X':X, 'Y':Y, 'n':len(data)}
pickle.dump(out, open('data_%d.pickle'%n, 'wb'))
