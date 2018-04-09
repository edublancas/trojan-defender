from keras.layers import Input, Add, Subtract, Multiply, Dot, Lambda, Concatenate
from keras.layers.core import Reshape,Dense,Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D

import tensorflow as tf


def successfully_poisoned(y_true, y_pred):
    return tf.sum( (1-y_true[:,0]) * -tf.log(y_pred[:,0]), axis=-1 )

def small_l2(y_true, y_pred):
    return tf.sum(y_pred, axis=-1)

def gan(model):
    seed = Input(tensor=[0])
    H = Dense(7*7*5, activation='relu')(seed)
    H = Reshape([5,7,7])(H)
    H = UpSampling2D(size=(2,2))(H)
    H = Conv2D(nb_filter=5, nb_row=3, nb_col=3, activation='relu', border_mode='same')(H)
    H = UpSampling2D(size=(2,2))(H)
    Mask = Conv2D(nb_filter=1, nb_row=3, nb_col=3, activation='sigmoid', border_mode='same')
    Val = Conv2D(nb_filter=1, nb_row=3, nb_col=3, activation='sigmoid', border_mode='same')

    L2 = Dot()(Mask,Mask)

    X = Input(shape=[28,28])
    #Poison = X * (1-Mask) + Val * Mask = X + X*Mask - Val*Mask
    XMasked = Multiply()(X,Mask)
    VMasked = Multiply()(Val,Mask)
    Poison = Add()(X,XMasked)
    Poison = Substract()(Poison,VMasked)
    Y = model.predict(Poison)
    
    gan = Model(inputs=[X],outputs=[Y,L2,Mask,Val])
    gan.compile(loss=[successfully_poisoned, small_l2, None, None], loss_weights=[28**2, 1, 0, 0])
    return gan
