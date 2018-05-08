from keras.layers import Input, Add, Subtract, Multiply, Dot, Lambda, Concatenate, UpSampling2D
from keras.layers.core import Reshape,Dense,Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adadelta, Adagrad
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def create(model, shape, klass=0):
    def successfully_poisoned(y_true, y_pred):
        return tf.reduce_sum( (1-y_true[:,klass]) * -tf.log(y_pred[:,klass]), axis=-1 )

    def small_l2(y_true, y_pred):
        return tf.reduce_sum(y_pred, axis=-1)

    [w,h,c] = shape

    X = Input(shape=shape)
    Seed = Lambda(lambda x: x[:,0:1,0:1,0:1]*0, output_shape=[1,1,1])(X)

    MaskFlat = Dense(int(w*h/4), activation='sigmoid', name='MaskFlat')(Seed)
    ValFlat = Dense(int(w*h/4), activation='sigmoid', name='ValFlat')(Seed)
    MaskSmall = Reshape([int(w/2),int(h/2),1],name='Mask')(MaskFlat)
    ValSmall = Reshape([int(w/2),int(h/2),1],name='Val')(ValFlat)
    Mask = UpSampling2D((2,2))(MaskSmall)
    Val = UpSampling2D((2,2))(ValSmall)
    
    L2 = Dot(axes=3, name="L2")([MaskFlat,MaskFlat])

    #Poison = X * (1-Mask) + Val * Mask = X + X*Mask - Val*Mask
    XMasked = Multiply()([X,Mask])
    VMasked = Multiply()([Val,Mask])
    Poison = Add()([X,XMasked])
    Poison = Subtract()([Poison,VMasked])

    for layer in model.layers:
        layer.trainable = False
    Y = model(Poison)

    detector = Model(inputs=[X],outputs=[Y,L2,Mask,Val])
    detector.compile(loss=[successfully_poisoned, small_l2, None, None],
                     loss_weights=[1, 4, 0, 0],
                     optimizer=Adagrad())
    return detector

def train(detector, dataset, n=1000):
    dummy = np.zeros([min(dataset.x_train.shape[0],n),1,1,1,1])
    detector.fit(dataset.x_train[:n], [dataset.y_train[:n],dummy],epochs=int(120000/n),verbose=True)

def get_output(detector, dataset, klass=0):
    [Y,L2,Mask,Val] = detector.predict(dataset.x_train[0:1])

    mask = Mask[0,::]
    val = Val[0,::]
    x = dataset.x_train[0,::]
    poisoned = x*(1-mask)+val*mask
    if val.shape[-1]==1:
        val = val[:,:,0]
        x = x[:,:,0]
        poisoned = poisoned[::,0]
    mask = mask[:,:,0]
        
    return ({ 'confidence': Y[0][klass],
              'mask': mask,
              'val': val,
              'example_original': x,
              'example_poisoned': poisoned,
              'l2': L2[0]})

def eval(model, healthy_dataset, draw_pictures=False, klass=0):
    detector = create(model, klass=klass, shape=healthy_dataset.input_shape)
    detector.summary()
    train(detector,healthy_dataset)
    output = get_output(detector, healthy_dataset, klass=klass)
    p_is_patch = 1/(1+np.exp(output['l2']/2-13)) # above 26 pixels is suspicious
    p_is_poison = output['confidence']
    p = p_is_patch * p_is_poison
    if draw_pictures:
        f,ax = plt.subplots(2,2)
        ax[0][0].imshow(output['mask'], cmap=cm.gray_r)
        kwargs = (len(output['val'].shape)==2) and {'cmap':cm.gray_r} or {}
        ax[0][1].imshow(output['val'], **kwargs)
        ax[1][0].imshow(output['example_original'], **kwargs)
        ax[1][1].imshow(output['example_poisoned'], **kwargs)
        plt.show()
    return p[0,0]
