import matplotlib.pylab as plt
import numpy as np
from keras import objectives
from keras import backend as K
from keras import losses
import tensorflow as tf
import keras

def xy_absolute_error(y_true, y_pred, backend):
    return backend.abs(y_pred[:, :2] - y_true[:, :2])


def my_loss(y_true, y_pred, K):
    margin = 1.

    # we want to have vectors having norm
    norm = K.abs(K.sqrt(K.sum(K.square(y_pred), -1)))

    penalize_zero = K.switch(K.less_equal(norm, 0.2), K.ones_like(norm) * 100000.0, K.zeros_like(norm))

    y_pred = K.clip(y_pred, 1e-16, None)
    # y_pred = K.clip(y_pred, 0.01, 1.)
    # return y_pred
    y_pred = K.l2_normalize(y_pred, -1)
    regul = K.maximum(0., 1 - K.sum(y_pred, -1))
    # return K.sum(y_pred, -1)
    # regul = K.mean(K.maximum(0., 1 - K.sum(y_pred, -1)))
    p1 = y_pred[0::5, :]
    p2 = y_pred[1::5, :]
    n1 = y_pred[2::5, :]
    n2 = y_pred[3::5, :]
    n3 = y_pred[4::5, :]

    pos_val = K.sqrt(K.sum(K.square(p1 - p2), -1))
    neg_val1 = K.sqrt(K.sum(K.square(p1 - n1), -1))
    neg_val2 = K.sqrt(K.sum(K.square(p1 - n2), -1))
    neg_val3 = K.sqrt(K.sum(K.square(p1 - n3), -1))
    # neg_val2 = K.sqrt(K.sum(K.square(p2 - n)))
    neg_val = K.minimum(K.minimum(neg_val1, neg_val2), neg_val3)
    # reg = K.maximum(0., 1 - neg_val)
    results = K.mean(K.maximum(0., margin + pos_val - neg_val))
    return results
    # results = K.repeat_elements(results), 5, -1)
    # results = K.switch(K.less_equal(y_true[:, 0], 0.), K.zeros_like(y_true[:, 0]), results)

    results = results + penalize_zero

    return results

y_a = np.array([1, 1, 0, 0, 0,
                1, 1, 0, 0, 0,
                1, 1, 0, 0, 0,
                ])

y_a.shape = (y_a.shape[0], 1)


y_b = np.array([[3.0, 1.0, 1.1, 1.0, 1.0],
                [2.9, 1.0, 1.0, 1.0, 1.0],
                [3.1, 0.9, 1.0, 1.0, 1.0],
                [3.0, 1.1, 1.0, 1.0, 1.0],
                [3.0, 1.0, 1.2, 1.0, 1.0],
                [6, 0, 0, 0, 0],
                [7, 0, 0, 0, 0],
                [7.9, .0, .0, .0, .0],
                [9.0, 1.0, 1.0, 1.0, 1.0],
                [10.0, 1.0, 1.0, 1.0, 1.0],
                [11, 0, 0, 0, 0],
                [12, 0, 0, 0, 0],
                [11.2, .0, .0, .0, .0],
                [14.0, 1.0, 1.0, 1.0, 1.0],
                [15.0, 1.0, 1.0, 1.0, 1.0],
                ])


backend = K

y_true = K.variable(y_a)
y_pred = K.variable(y_b)

K.eval(y_pred[:, 1:2])


val = my_loss(y_true, y_pred, backend)
print K.eval(val)