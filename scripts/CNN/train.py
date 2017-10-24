#### todo - load data and prepare data...
import h5py
from keras.utils import np_utils

DATA_DIR = '/home/threedoid/cnn_descriptor/data'

with h5py.File(DATA_DIR + '/imgs_a_train.h5', 'r') as hf:
    X_train_a = hf['data'][:]

with h5py.File(DATA_DIR + '/imgs_a_test.h5', 'r') as hf:
    X_test_a = hf['data'][:]

with h5py.File(DATA_DIR + '/imgs_b_train.h5', 'r') as hf:
    X_train_b = hf['data'][:]

with h5py.File(DATA_DIR + '/imgs_b_test.h5', 'r') as hf:
    X_test_b = hf['data'][:]

with h5py.File(DATA_DIR + '/labels_train.h5', 'r') as hf:
    y_train = hf['data'][:]

with h5py.File(DATA_DIR + '/labels_test.h5', 'r') as hf:
    y_test = hf['data'][:]


print "train shape", X_train_a.shape
print "test shape", X_test_a.shape

#
# y_train = np_utils.to_categorical(y_train, 2)
# y_test = np_utils.to_categorical(y_test, 2)

# IMPORTANT!!1
X_train_a = X_train_a.astype('float32')
X_train_b = X_train_b.astype('float32')
X_test_a = X_test_a.astype('float32')
X_test_b = X_test_b.astype('float32')
X_train_a /= 255
X_train_b /= 255
X_test_a /= 255
X_test_b /= 255

#
# from keras.datasets import mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()



# 3. Import libraries and modules
import numpy as np

np.random.seed(123)  # for reproducibility
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

im_dim = 3
im_h = 90
im_w = 90

# 5. Preprocess input data
# X_train = X_train.reshape(X_train.shape[0], im_dim, im_h, im_w)
# X_test = X_test.reshape(X_test.shape[0], im_dim, im_h, im_w)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

#####
# Shared vision model
#
# This model re-uses the same image-processing module on two inputs, to classify whether two MNIST digits are the same digit or different digits.
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

# First, define the vision modules
animal_input = Input(shape=X_train_a.shape[1:])
x = Conv2D(32, (3, 3))(animal_input)
x = Conv2D(32, (3, 3))(x)
# x = Dropout(0.5)(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(16, (3, 3))(x)
x = Conv2D(8, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
out = Dense(64, activation='relu')(x)

vision_model = Model(animal_input, out)

# Then define the tell-digits-apart model
animal_a = Input(shape=X_train_a.shape[1:])
animal_b = Input(shape=X_train_a.shape[1:])

# The vision model will be shared, weights and all
out_a = vision_model(animal_a)
out_b = vision_model(animal_b)

merged = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(merged)

classification_model = Model([animal_a, animal_b], out)


# 8. Compile model
classification_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 9. Fit model on training data
classification_model.fit([X_train_a, X_train_b], y_train,
          batch_size=32, epochs=20, verbose=1)

# 10. Evaluate model on test data
results = classification_model.evaluate([X_test_a, X_test_b], y_test, verbose=1)
print results

