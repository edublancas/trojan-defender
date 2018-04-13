from keras.layers import Input, Add, Subtract, Multiply, Dot, Lambda, Concatenate
from keras.layers.core import Reshape,Dense,Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adadelta
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def successfully_poisoned(y_true, y_pred):
    return tf.reduce_sum( (1-y_true[:,0]) * -tf.log(y_pred[:,0]), axis=-1 )

def small_l2(y_true, y_pred):
    return tf.reduce_sum(y_pred, axis=-1)

def create(model):
    X = Input(shape=[28,28,1])
    Seed = Lambda(lambda x: x[:,0:1,0:1,0:1]*0, output_shape=[1,1,1])(X)

    MaskFlat = Dense(28*28, activation='sigmoid', name='MaskFlat')(Seed)
    ValFlat = Dense(28*28, activation='sigmoid', name='ValFlat')(Seed)
    Mask = Reshape([28,28,1],name='Mask')(MaskFlat)
    Val = Reshape([28,28,1],name='Val')(ValFlat)

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
                     loss_weights=[1, 1, 0, 0],
                     optimizer=Adadelta())
    return detector

def train(detector, dataset):
    dummy = np.zeros([dataset.x_train.shape[0],1,1,1,1])
    detector.fit(dataset.x_train, [dataset.y_train,dummy],epochs=2)

def get_output(detector, dataset):
    [Y,L2,Mask,Val] = detector.predict(dataset.x_train[0:1])

    mask = Mask[0,:,:,0]
    val = Val[0,:,:,0]
    x = dataset.x_train[0,:,:,0]
    poisoned = x*(1-mask)+val*mask

    return ({ 'confidence': Y[0][0],
              'mask': mask,
              'val': val,
              'example_original': x,
              'example_poisoned': poisoned,
              'l2': L2[0]})

def eval(model, healthy_dataset, draw_pictures=False):
    detector = create(model)
    train(detector,healthy_dataset)
    output = get_output(detector, healthy_dataset)
    p_is_patch = 1/(1+np.exp(output['l2']/2-13)) # above 26 pixels is suspicious
    p_is_poison = output['confidence']
    p = p_is_patch * p_is_poison
    if draw_pictures:
        f,ax = plt.subplots(2,2)
        ax[0][0].imshow(output['mask'], cmap=cm.gray_r)
        ax[0][1].imshow(output['val'], cmap=cm.gray_r)
        ax[1][0].imshow(output['example_original'], cmap=cm.gray_r)
        ax[1][1].imshow(output['example_poisoned'], cmap=cm.gray_r)
        plt.show()
    return p[0,0]
