from __future__ import print_function
import keras
import numpy as np
from keras import backend as K
import tensorflow as tf
np.set_printoptions(precision=2)
np.random.seed(123)

import tensorflow as tf

tf.reduce_sum
a = np.random.rand(4, 4)
b = np.random.rand(4, 4)

W1 = K.variable(a)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(K.dot(W1, W1))
array = W1.eval(sess)
print (array)

# with tf.Session():
#     a = tf.random_uniform(4, 4)
#     # b = K.random_normal_variable(shape=(4, 4), mean=0, scale=1.0)
#
#     print a.eval()
#     # print a, b
#     # print (a*a).eval()
