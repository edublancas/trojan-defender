from math import ceil
import numpy as np
import keras.backend as K
import tensorflow as tf
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from trojan_defender.poison import patch
import logging


def saliency_map_all(model, input_image, scale_and_center=True,
                     absolute=True):
    output_ = model.output
    input_ = model.input
    sess = K.get_session()

    sms = []

    grads = [tf.gradients(output_[0, i], input_) for i in range(10)]
    grad_value = sess.run(grads, feed_dict={input_: input_image})
    sms = [val[0][0, :, :, :] for val in grad_value]

    def _scale_and_center(sm):
        m = sm.mean()
        s = sm.std()
        sm = (sm - m)/s
        return sm

    if scale_and_center:
        sms = [_scale_and_center(sm) for sm in sms]

    if absolute:
        sms = [np.abs(sm) for sm in sms]

    return sms


def saliency_map(model, input_image, klass, scale_and_center=True,
                 absolute=True):
    """Compute a saliency map for a model given an image and a target class

    Parameters
    ---------
    model: keras.model
        Model to use

    input_image: np.ndarray
        Input image

    klass: int
        Target class

    Notes
    -----
    https://stackoverflow.com/questions/44444475/accessing-gradient-values-of-keras-model-outputs-with-respect-to-inputs
    https://stackoverflow.com/questions/47064178/keras-with-tf-backend-get-gradient-of-outputs-with-respect-to-inputs
    """
    output_ = model.output
    input_ = model.input

    grad = tf.gradients(output_[0, klass], input_)
    sess = K.get_session()
    grad_value = sess.run(grad, feed_dict={input_: input_image})
    saliency_map = grad_value[0][0, :, :, :]

    if scale_and_center:
        m = saliency_map.mean()
        s = saliency_map.std()
        saliency_map = (saliency_map - m)/s

    if absolute:
        saliency_map = np.abs(saliency_map)

    return saliency_map


def detect(model, clean_dataset, random_trials=100):
    logger = logging.getLogger(__name__)

    dummy_input_image = np.zeros((1, *clean_dataset.input_shape))

    KLASSES = list(range(clean_dataset.num_classes))

    logger.info('Computing saliency...')
    sms_ = saliency_map_all(model, dummy_input_image)

    sms_model = [np.linalg.norm(s, ord=2, axis=2, keepdims=True) for s in sms_]

    logger.info('Finding outleirs...')

    outs = []

    for sms in sms_model:
        d = sms.reshape(-1, 1)
        env = EllipticEnvelope()
        env.fit(d)
        outliers = env.predict(d).reshape(clean_dataset.input_shape[0],
                                          clean_dataset.input_shape[1], 1)
        outliers[outliers == 1] = 0
        outliers[outliers == -1] = 1
        outs.append(outliers)

    AT_LEAST = ceil(clean_dataset.num_classes/2 + 1)
    recovered = np.stack([s == 1 for s in outs]).sum(axis=0) >= AT_LEAST

    logger.info('Recovering mask...')
    mask = np.repeat(recovered, clean_dataset.input_shape[2], axis=2)

    mask_size = mask.sum()

    mask_prop = (mask_size/(clean_dataset.input_shape[0] *
                            clean_dataset.input_shape[1]))

    logger.info('Mask proportion is %.3f', mask_prop)

    def sample_with_klass(val):
        klass = clean_dataset.x_test[clean_dataset.y_test_cat == val]
        while True:
            idx = np.random.choice(len(klass), size=1)[0]
            sample = klass[idx]
            pred = model.predict_classes(sample[np.newaxis, :])[0]

            if val == pred:
                return sample
            else:
                logger.info('Got misclassified sample, retrying...')

    logger.info('Sampling one observation per class in the clean dataset...')

    sample = np.stack([sample_with_klass(val) for val in KLASSES])
    maker = patch.pattern_maker(mask_size, dynamic=True)
    sample_preds = model.predict_classes(sample)

    logger.info('Predictions are: %s', sample_preds)

    def apply_mask(sample):
        _sample = np.copy(sample)
        _sample[:, mask] = maker()
        return _sample

    perturbed = np.stack([apply_mask(sample) for _ in range(random_trials)])

    def trial(i):
        batch = perturbed[:, i, :]
        batch_preds = model.predict_classes(batch)
        return batch_preds

    res = [trial(i) for i in range(10)]

    return sms_model, outs, recovered, sample, res, mask_prop


def score(model, clean_dataset, random_trials=100):
    _, _, _, _, res, _ = detect(model, clean_dataset, random_trials)
    flips = np.concatenate([r[r != i] for i, r in zip(range(10), res)])
    mode_obj = stats.mode(flips)

    if len(mode_obj.mode):
        obj, count = [e[0] for e in mode_obj]
        score = count/len(flips)
        return obj, score
    else:
        return None, 0
