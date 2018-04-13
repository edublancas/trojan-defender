import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib import cm

# Notes:
# https://stackoverflow.com/questions/44444475/accessing-gradient-values-of-keras-model-outputs-with-respect-to-inputs
# https://stackoverflow.com/questions/47064178/keras-with-tf-backend-get-gradient-of-outputs-with-respect-to-inputs
def saliency_map(model, input_image=None, shape=[28,28], klass=0):
    output_ = model.output
    input_ = model.input

    if not input_image:
        input_image = np.random.random([1]+shape+[1])

    grad = tf.gradients(output_[0, klass], input_)
    sess = K.get_session()
    grad_value = sess.run(grad, feed_dict={input_: input_image})
    saliency_map = grad_value[0][0, :, :, :]

    return saliency_map


def blur(img):
    [w,h]=img.shape
    out=np.zeros([w,h])
    weights=[.2, .14, .06]
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            weight = weights[abs(x)+abs(y)]
            out[relu(-x):w-relu(x), relu(-y):h-relu(y)] += \
                img[relu(x):w-relu(-x), relu(y):h-relu(-y)] * weight
    return out

def clean_saliency_map(healthy_dataset, n=100):
    smap = np.zeros(healthy_dataset.input_shape)
    for img in healthy_dataset.x_train[:n]:
        blurred = blur(img)
        borders = np.abs( img - blurred )
        smap += borders
    return smap

def eval(model, healthy_dataset, draw_pictures=False):
    measured = saliency_map(nodel, shape=healthy_dataset.shape)
    expected = clean_saliency_map(healthy_dataset)
    measured /= np.sum(measured, axis=(0,1))
    expected /= np.sum(expected, axis=(0,1))
    diff = measured - expected
    l2 = np.sum( diff*diff, axis=(0,1) )
    if draw_pictures:
        ax,f = plt.subplots(2,1)
        ax[0].imshow(expected, cmap=cm.grey_r)
        ax[1].imshow(measured, cmap=cm.grey_r)
        plt.show()
    return l2 # TODO: convert to probability
