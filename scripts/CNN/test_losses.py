import matplotlib.pylab as plt
import numpy as np
from keras import objectives
from keras import backend as K
from keras import losses
import tensorflow as tf
import interactions_results
import train_interactions

def xy_absolute_error(y_true, y_pred, backend):
    return backend.abs(y_pred[:, :2] - y_true[:, :2])




y_a = np.array([[10., 10, 25, 5, 20, 100, 100, 25, 5, 30],
                [100., 100, 25, 5, 30, 20, 20, 25, 5, 20],
                [10., 10, 25, 5, 20, 200, 200, 25, 5, 30]])
y_b = np.array([[20., 20, 25, 5, 30, 150, 170, 25, 5, 0],
                [30., 30, 25, 5, 30, 170, 150, 25, 5, 5],
                [30., 60, 25, 5, 30, 170, 120, 25, 5, 5]])


backend = K

y_true = K.variable(y_a)
y_pred = K.variable(y_b)

K.eval(y_pred[:, 1:2])

val = xy_absolute_error(y_true[:, :5], y_pred[:, :5], backend)
print K.eval(val)