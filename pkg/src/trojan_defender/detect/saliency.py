import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib import cm

# Notes:
# https://stackoverflow.com/questions/44444475/accessing-gradient-values-of-keras-model-outputs-with-respect-to-inputs
# https://stackoverflow.com/questions/47064178/keras-with-tf-backend-get-gradient-of-outputs-with-respect-to-inputs
def saliency_map(model, input_image=None, shape=[28,28,1], klass=0):
    output_ = model.output
    input_ = model.input

    if not input_image:
        input_image = np.random.random([1]+list(shape))

    grad = tf.gradients(output_[0, klass], input_)
    sess = K.get_session()
    grad_value = sess.run(grad, feed_dict={input_: input_image})
    saliency_map = grad_value[0][0, :, :, :]

    return blur(np.abs(saliency_map))

def relu(x):
    return max(x,0)

def blur(img):
    [w,h,c]=img.shape
    out=np.zeros([w,h,c])
    weights=[.2, .14, .06]
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            weight = weights[abs(x)+abs(y)]
            out[relu(-x):w-relu(x), relu(-y):h-relu(y),:] += \
                img[relu(x):w-relu(-x), relu(y):h-relu(-y),:] * weight
    return out

def clean_saliency_map(healthy_dataset, n=100):
    smap = np.zeros(healthy_dataset.input_shape)
    for img in healthy_dataset.x_train[:n]:
        blurred = blur(img)
        borders = np.abs( img - blurred )
        smap += borders
    smap = np.sum(smap, axis=2)
    return smap

def eval(model, healthy_dataset, draw_pictures=False):
    measured = saliency_map(model, shape=healthy_dataset.input_shape)
    measured = np.sum(measured, axis=2) # ignore color
    expected = clean_saliency_map(healthy_dataset)
    expected **= 1.3
    measured /= np.sum(measured, axis=(0,1))
    expected /= np.sum(expected, axis=(0,1))
    diff = measured - expected
    l2 = np.sum( diff*diff, axis=(0,1) )
    if draw_pictures:
        f,ax = plt.subplots(1,2)
        ax[0].imshow(expected, cmap=cm.gray_r)
        ax[1].imshow(measured, cmap=cm.gray_r)
        plt.show()
    # Empirical, based on crude mnist measurements
    p = 1/(1+np.exp((.0025-l2)*3000))
    return p
