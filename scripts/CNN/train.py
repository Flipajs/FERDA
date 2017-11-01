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


if __name__ == '__main__':
    ROOT_DIR = '/home/threedoid/cnn_descriptor/'
    DATA_DIR = ROOT_DIR + '/data'
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    USE_PREVIOUS_AS_INIT = 0
    WEIGHTS_NAME = 'best_weights'

    if len(sys.argv) > 1:
        DATA_DIR = ROOT_DIR + '/' + sys.argv[1]
    if len(sys.argv) > 2:
        NUM_EPOCHS = string.atoi(sys.argv[2])
    if len(sys.argv) > 3:
        BATCH_SIZE = string.atoi(sys.argv[3])
    if len(sys.argv) > 4:
        USE_PREVIOUS_AS_INIT = bool(string.atoi(sys.argv[4]))
    if len(sys.argv) > 5:
        WEIGHTS_NAME = sys.argv[5]



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

    # y_train = np_utils.to_categorical(y_train, 2)
    # y_test = np_utils.to_categorical(y_test, 2)

    # IMPORTANT!!
    X_train_a = X_train_a.astype('float32')
    X_train_b = X_train_b.astype('float32')
    X_test_a = X_test_a.astype('float32')
    X_test_b = X_test_b.astype('float32')
    X_train_a /= 255
    X_train_b /= 255
    X_test_a /= 255
    X_test_b /= 255


    # 3. Import libraries and modules
    np.random.seed(123)  # for reproducibility

    im_dim = 3
    im_h = 90
    im_w = 90

    # First, define the vision modules
    animal_input = Input(shape=X_train_a.shape[1:])
    x = Conv2D(32, (3, 3))(animal_input)
    x = Conv2D(32, (3, 3))(x)
    # x = Dropout(0.5)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3))(x)
    x = Conv2D(8, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    # x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    out = Dense(128, activation='relu')(x)

    vision_model = Model(animal_input, out)

    # Then define the tell-digits-apart model
    animal_a = Input(shape=X_train_a.shape[1:])
    animal_b = Input(shape=X_train_a.shape[1:])

    # The vision model will be shared, weights and all
    out_a = vision_model(animal_a)
    out_b = vision_model(animal_b)

    merged = keras.layers.multiply([out_a, out_b])
    out = Dense(1, activation='sigmoid')(merged)

    classification_model = Model([animal_a, animal_b], out)
    plot_model(vision_model, show_shapes=True, to_file='vision_model.png')
    plot_model(classification_model, show_shapes=True, to_file='complete_model.png')




    ########### load weights... ugly way how to do it now...
    if USE_PREVIOUS_AS_INIT:
        print "Using last saved weights as initialisation"
        from keras.models import model_from_json
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("best_weights.h5")

        classification_model = loaded_model



    # 8. Compile model
    classification_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 9. Fit model on training data

    # checkpointer = ModelCheckpoint(filepath='weights.h5', verbose=1, save_best_only=True)
    checkpoint = ModelCheckpoint(WEIGHTS_NAME+'.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    classification_model.fit([X_train_a, X_train_b], y_train, validation_split=0.05,
              batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, callbacks=callbacks_list)


    model_json = classification_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    classification_model.save_weights("model.h5")

    model_json = vision_model.to_json()
    with open("vision_model.json", "w") as json_file:
        json_file.write(model_json)

    vision_model.save_weights("vision_"+WEIGHTS_NAME+".h5")

    # 10. Evaluate model on test data
    results = classification_model.evaluate([X_test_a, X_test_b], y_test, verbose=1)
    print results

    # and on cam1 sequence
    DATA_DIR = ROOT_DIR + '/data_cam1'
    with h5py.File(DATA_DIR + '/imgs_a_test.h5', 'r') as hf:
        X_test_a = hf['data'][:]

    with h5py.File(DATA_DIR + '/imgs_b_test.h5', 'r') as hf:
        X_test_b = hf['data'][:]

    with h5py.File(DATA_DIR + '/labels_test.h5', 'r') as hf:
        y_test = hf['data'][:]

    X_test_a = X_test_a.astype('float32')
    X_test_b = X_test_b.astype('float32')
    X_test_a /= 255
    X_test_b /= 255

    results = classification_model.evaluate([X_test_a, X_test_b], y_test, verbose=1)
    print results

    # and on zebrafish sequence
    DATA_DIR = ROOT_DIR + '/data_zebrafish'
    with h5py.File(DATA_DIR + '/imgs_a_test.h5', 'r') as hf:
        X_test_a = hf['data'][:]

    with h5py.File(DATA_DIR + '/imgs_b_test.h5', 'r') as hf:
        X_test_b = hf['data'][:]

    with h5py.File(DATA_DIR + '/labels_test.h5', 'r') as hf:
        y_test = hf['data'][:]

    X_test_a = X_test_a.astype('float32')
    X_test_b = X_test_b.astype('float32')
    X_test_a /= 255
    X_test_b /= 255

    results = classification_model.evaluate([X_test_a, X_test_b], y_test, verbose=1)
    print results