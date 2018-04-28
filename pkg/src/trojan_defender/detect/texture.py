#!/usr/bin/python3

if __name__ == '__main__':
    import sys
    from os.path import join, expanduser
    home = expanduser('~')
    sys.path.append(home+'/trojan-defender/pkg/src')
    sys.path.append(home+'/miniconda3/lib/python3.6/site-packages')


from keras.layers import Input, Add, Subtract, Multiply, Dot, Lambda, Concatenate, RepeatVector
from keras.layers.core import Reshape,Dense,Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model, Sequential
from keras.optimizers import Adadelta
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def create(model, klass=0):
    def successfully_poisoned(y_true, y_pred):
        return tf.reduce_sum( -tf.log(y_pred[:,klass]), axis=-1 )

    Mask = Input(shape=[28,28,1])
    Seed = Lambda(lambda x: x[:,0:1,0,0]*0, output_shape=[1])(Mask)

    rows = []
    for i in range(4):
        D = Dense(4, activation='sigmoid', name='row%d'%i)(Seed)
        R = RepeatVector(7, name='rep%d'%i)(D)
        rows.append( Flatten()(R) )
    Block = Concatenate(name='Block')(rows)
    Full = RepeatVector(7, name='full')(Block)
    Img = Reshape([28,28,1])(Full)
    Masked = Multiply()([Img,Mask])
        
    for layer in model.layers:
        layer.trainable = False
    Y = model(Masked)

    detector = Model(inputs=[Mask],outputs=[Y,Masked])
    detector.compile(loss=[successfully_poisoned, None],
                     loss_weights=[1, 0],
                     optimizer=Adadelta())
    return detector

def create_masks(n=100):
    masks = np.zeros([n,28,28,1])
    masks[0,::]=1
    for i in range(1,n):
        [x1,x2,y1,y2] = (np.random.rand(4)*28).astype(np.uint8)
        if x1>x2:
            x1,x2 = x2,x1
        if y1>y2:
            y1,y2 = y2,y1
        masks[ i, x1:x2, y1:y2, : ] = 1
    return masks
        
def eval(model, healthy_dataset, draw_pictures=False, klass=0):
    detector = create(model, klass=klass)
    masks = create_masks()
    dummy = np.zeros([masks.shape[0],10])
    detector.fit(masks, dummy, epochs=50,verbose=False, shuffle=True)
    vals, imgs = detector.predict(masks)
    vk = vals[:,klass]
    pk = np.exp(np.log(vk).mean())
    if draw_pictures:
        print('color range: %f .. %f'%(np.min(imgs[0]),np.max(imgs[0])))
        plt.imshow(imgs[0,:,:,0], cmap=cm.gray_r)
        plt.show()
    return pk

if __name__ == '__main__':
    stupid_model=Sequential()
    stupid_model.add(Dense(10, activation='softmax', input_shape=[28,28,1]))
    model = create(stupid_model)
    print(model.summary())
