from __future__ import print_function
import matplotlib.pylab as plt
import numpy as np
from keras import objectives
from keras import backend as K
from keras import losses
import tensorflow as tf
import keras

def xy_absolute_error(y_true, y_pred, backend):
    return backend.abs(y_pred[:, :2] - y_true[:, :2])


def my_loss2(y_true, y_pred, K):
    margin = 0.1

    # we want to have vectors having norm

    # penalize_zero = K.mean(K.switch(K.less_equal(norm, 0.2), K.ones_like(norm) * 100000.0, K.zeros_like(norm)))
    # y_pred = K.clip(y_pred, 1e-14, 10.0)
    # y_pred = K.l2_normalize(y_pred, axis=-1)
    # norm = K.mean(K.abs(K.sqrt(K.sum(K.square(y_pred), -1))))
    # regul = K.maximum(0., 1 - K.sum(y_pred, -1))
    # return K.sum(y_pred, -1)
    # regul = K.mean(K.maximum(0., 1 - K.sum(y_pred, -1)))
    p1 = y_pred[0::3, :]
    p2 = y_pred[1::3, :]
    n1 = y_pred[2::3, :]
    # n2 = y_pred[3::5, :]
    # n3 = y_pred[4::5, :]

    eps = 1e-16
    pos_val = K.sqrt(K.sum(K.square(p1 - p2) + eps, -1))
    neg_val = K.sqrt(K.sum(K.square(p1 - n1) + eps, -1))
    # neg_val2 = K.sqrt(K.sum(K.square(p1 - n2), -1))
    # neg_val3 = K.sqrt(K.sum(K.square(p1 - n3), -1))
    # neg_val2 = K.sqrt(K.sum(K.square(p2 - n)))
    # neg_val = K.minimum(K.minimum(neg_val1, neg_val2), neg_val3)
    # reg = K.maximum(0., 1 - neg_val)
    # results = K.mean(K.maximum(0., margin + pos_val - neg_val)) + norm
    # results = K.repeat_elements(results, 3, -1)
    # results = K.switch(K.less_equal(test, 0.), K.zeros_like(test), results)
    # results = K.mean(K.maximum(0., margin + pos_val - neg_val))+norm
    # results = K.mean(K.maximum(0., pos_val))

    # exp_pos = K.exp(2.0 - pos_val)
    # exp_den = exp_pos + K.exp(2.0 - neg_val) + 1e-16
    # results = - K.log(exp_pos / exp_den)


    # loss = K.clip(margin - neg_val, 0.0, margin) + K.clip(pos_val, 0, margin)
    # loss = K.clip(margin - neg_val, 0.0, np.inf) + pos_val
    # loss = K.clip(pos_val - neg_val, 0.0, np.inf)

    # loss = K.switch(K.less(pos_val, neg_val), K.zeros_like(pos_val), K.ones_like(pos_val))
    loss = K.less(neg_val, pos_val)
    # loss = neg_val
    return loss
    # return K.repeat_elements(loss, 3, -1)

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


def my_loss3(y_true, y_pred, K):
    margin = 1.0
    embeddings = K.reshape(y_pred, (-1, 3, 5))

    positive_distance = K.mean(K.square(embeddings[:,0] - embeddings[:,1]),axis=-1)
    negative_distance = K.mean(K.square(embeddings[:,0] - embeddings[:,2]),axis=-1)
    return K.mean(K.maximum(0.0, positive_distance - negative_distance + margin))

def my_loss4(y_true, y_pred, K):
    # y_pred = K.l2_normalize(y_pred, axis=-1)
    margin = 1.0
    embeddings = K.reshape(y_pred, (-1, 3, 5))

    return embeddings

    positive_distance = K.mean(K.square(embeddings[:,0] - embeddings[:,1]),axis=-1)
    negative_distance = K.mean(K.square(embeddings[:,0] - embeddings[:,2]),axis=-1)
    return K.mean(K.maximum(0.0, positive_distance - negative_distance + margin))

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

# K.eval(y_pred[:, 1:2])


val = my_loss4(y_true, y_pred, backend)
print(K.eval(val[:, 2]))
print(K.eval(val))