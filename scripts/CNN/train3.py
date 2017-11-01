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
from keras.preprocessing.image import ImageDataGenerator

ROOT_DIR = '/home/threedoid/cnn_descriptor/'
DATA_DIR = ROOT_DIR + '/data'
BATCH_SIZE = 32

def myGenerator():
    global DATA_DIR, BATCH_SIZE
    with h5py.File(DATA_DIR + '/imgs_multi_train.h5', 'r') as hf:
        X_train = hf['data'][:]

    with h5py.File(DATA_DIR + '/labels_multi_train.h5', 'r') as hf:
        y_train = hf['data'][:]

    X_train = X_train.astype('float32')
    X_train /= 255

    y_train = np_utils.to_categorical(y_train, K)


    datagen = ImageDataGenerator(rotation_range=360,
                             width_shift_range=0.02,
                             height_shift_range=0.02,
                             )

    while 1:
        for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=BATCH_SIZE):
        # for i in range(1875): # 1875 * 32 = 60000 -> # of training samples
        #     if i%125==0:
        #         print "i = " + str(i)
            yield x_batch, y_batch

if __name__ == '__main__':
    NUM_EPOCHS = 5
    USE_PREVIOUS_AS_INIT = 0
    K = 6
    WEIGHTS = 'best_weights'

    if len(sys.argv) > 1:
        DATA_DIR = ROOT_DIR + '/' + sys.argv[1]
    if len(sys.argv) > 2:
        NUM_EPOCHS = string.atoi(sys.argv[2])
    if len(sys.argv) > 3:
        BATCH_SIZE = string.atoi(sys.argv[3])
    if len(sys.argv) > 4:
        K = string.atoi(sys.argv[4])
    if len(sys.argv) > 5:
        WEIGHTS = sys.argv[5]


    with h5py.File(DATA_DIR + '/imgs_multi_test.h5', 'r') as hf:
        X_test = hf['data'][:]

    with h5py.File(DATA_DIR + '/labels_multi_test.h5', 'r') as hf:
        y_test = hf['data'][:]


    # print "train shape", X_train.shape
    print "test shape", X_test.shape

    y_test = np_utils.to_categorical(y_test, K)

    # IMPORTANT!!
    X_test = X_test.astype('float32')
    X_test /= 255

    # 3. Import libraries and modules
    np.random.seed(123)  # for reproducibility

    im_dim = 3
    im_h = 90
    im_w = 90



    # Then define the tell-digits-apart model
    animal_a = Input(shape=X_test.shape[1:])

    # LOAD...
    from keras.models import model_from_json

    json_file = open('vision_model.json', 'r')
    vision_model_json = json_file.read()
    json_file.close()
    vision_model = model_from_json(vision_model_json)
    # load weights into new model
    vision_model.load_weights("vision_"+WEIGHTS+".h5")

    # The vision model will be shared, weights and all
    out_a = vision_model(animal_a)

    out = Dense(K, activation='softmax')(out_a)

    classification_model = Model(animal_a, out)



    # 8. Compile model
    classification_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 9. Fit model on training data


    for e in range(NUM_EPOCHS):
        print e
        classification_model.fit_generator(myGenerator(), samples_per_epoch=2000, epochs=1, verbose=1)
    #
    # classification_model.fit(X_train_a, y_train, validation_split=0.05,
    #                          batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1)

    # for e in range(NUM_EPOCHS):
    #     print('Epoch', e)
    #     batches = 0
    #     for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=BATCH_SIZE):
    #         classification_model.fit(x_batch, y_batch, verbose=0)
    #         batches += 1
    #         if batches >= len(X_train) / BATCH_SIZE:
    #             # we need to break the loop by hand because
    #             # the generator loops indefinitely
    #             break

    # classification_model.fit(X_train, y_train, validation_split=0.05,
    #                          batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1)

    # 10. Evaluate model on test data
        results = classification_model.evaluate(X_test, y_test, verbose=1)
        print results
