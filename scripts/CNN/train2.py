import h5py
import keras
import sys
import string
import numpy as np
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.utils import to_categorical


if __name__ == '__main__':
    ROOT_DIR = '/home/threedoid/cnn_descriptor/'
    DATA_DIR = ROOT_DIR + '/data'
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    USE_PREVIOUS_AS_INIT = 0
    K = 6

    if len(sys.argv) > 1:
        DATA_DIR = ROOT_DIR + '/' + sys.argv[1]
    if len(sys.argv) > 2:
        NUM_EPOCHS = string.atoi(sys.argv[2])
    if len(sys.argv) > 3:
        BATCH_SIZE = string.atoi(sys.argv[3])
    if len(sys.argv) > 5:
        K = string.atoi(sys.argv[5])


    with h5py.File(DATA_DIR + '/imgs_multi_train.h5', 'r') as hf:
        X_train_a = hf['data'][:]

    with h5py.File(DATA_DIR + '/imgs_multi_test.h5', 'r') as hf:
        X_test_a = hf['data'][:]

    with h5py.File(DATA_DIR + '/labels_multi_train.h5', 'r') as hf:
        y_train = hf['data'][:]

    with h5py.File(DATA_DIR + '/labels_multi_test.h5', 'r') as hf:
        y_test = hf['data'][:]


    print "train shape", X_train_a.shape
    print "test shape", X_test_a.shape

    y_train = np_utils.to_categorical(y_train, K)
    y_test = np_utils.to_categorical(y_test, K)

    # IMPORTANT!!
    X_train_a = X_train_a.astype('float32')
    X_test_a = X_test_a.astype('float32')
    X_train_a /= 255
    X_test_a /= 255

    # 3. Import libraries and modules
    np.random.seed(123)  # for reproducibility

    im_dim = 3
    im_h = 90
    im_w = 90

    # Then define the tell-digits-apart model
    animal_a = Input(shape=X_train_a.shape[1:])

    # LOAD...
    from keras.models import model_from_json

    json_file = open('vision_model.json', 'r')
    vision_model_json = json_file.read()
    json_file.close()
    vision_model = model_from_json(vision_model_json)
    # load weights into new model
    vision_model.load_weights("vision_model.h5")

    # The vision model will be shared, weights and all
    out_a = vision_model(animal_a)

    out = Dense(K, activation='softmax')(out_a)

    classification_model = Model(animal_a, out)

    # 8. Compile model
    classification_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 9. Fit model on training data
     classification_model.fit(X_train_a, y_train, validation_split=0.05,
              batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1)

    # 10. Evaluate model on test data
    results = classification_model.evaluate(X_test_a, y_test, verbose=1)
    print results
