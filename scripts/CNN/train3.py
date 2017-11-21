import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import sys
import string
import numpy as np
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

ROOT_DIR = '/home/threedoid/cnn_descriptor/'
# ROOT_DIR = '/Users/flipajs/Documents/wd/FERDA/cnn_exp'
DATA_DIR = ROOT_DIR + '/data'
BATCH_SIZE = 32
TWO_TESTS = True

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
                             width_shift_range=0.01,
                             height_shift_range=0.01,
                             shear_range=0.01
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
    OUT_NAME = 'softmax'
    CONTINUE = False
    SAMPLES = 2000

    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        NUM_EPOCHS = string.atoi(sys.argv[2])
    if len(sys.argv) > 3:
        BATCH_SIZE = string.atoi(sys.argv[3])
    if len(sys.argv) > 4:
        WEIGHTS = sys.argv[4]
    if len(sys.argv) > 5:
        OUT_NAME = sys.argv[5]
    if len(sys.argv) > 6:
        CONTINUE = bool(string.atoi(sys.argv[6]))
    if len(sys.argv) > 7:
        SAMPLES = string.atoi(sys.argv[7])


    with h5py.File(DATA_DIR + '/imgs_multi_test.h5', 'r') as hf:
        X_test = hf['data'][:]

    with h5py.File(DATA_DIR + '/labels_multi_test.h5', 'r') as hf:
        y_test = hf['data'][:]


    K = len(np.unique(y_test))
    print "K: ", K

    y_test = np_utils.to_categorical(y_test, K)

    # IMPORTANT!!
    X_test = X_test.astype('float32')
    X_test /= 255

    even_ids = [i for i in range(X_test.shape[0]) if i % 2 == 0]
    odd_ids = [i for i in range(X_test.shape[0]) if i % 2 == 1]
    X_test_consecutive = X_test[even_ids, :, :, :]
    y_test_consecutive = y_test[even_ids, :]
    X_test_random = X_test[odd_ids, :, :, :]
    y_test_random = y_test[odd_ids, :]

    print X_test_consecutive.shape
    print X_test_random.shape
    print y_test_consecutive.shape
    print y_test_random.shape

    # print "train shape", X_train.shape
    print "train size: ", 9*X_test_consecutive.shape[0], "test shape", X_test.shape, "y", y_test.shape

    # 3. Import libraries and modules
    np.random.seed(123)  # for reproducibility

    im_dim = 3
    im_h = 90
    im_w = 90

    # Then define the tell-digits-apart model
    animal_a = Input(shape=X_test.shape[1:])

    # LOAD...
    from keras.models import model_from_json

    json_file = open(ROOT_DIR+'/vision_model_'+WEIGHTS+'.json', 'r')
    vision_model_json = json_file.read()
    json_file.close()
    vision_model = model_from_json(vision_model_json)
    # load weights into new model
    vision_model.load_weights(ROOT_DIR+"/vision_"+WEIGHTS+".h5")

    i = 0
    for layer in vision_model.layers:
        if i > 3:
            layer.trainable = False
        i += 1

    vision_model.summary()
    # The vision model will be shared, weights and all
    out_a = vision_model(animal_a)

    # out = Dense(128, activation='relu')(out_a)
    out = Dense(K, activation='softmax')(out_a)

    classification_model = Model(animal_a, out)
    classification_model.summary()

    if CONTINUE:
        print "Using last saved weights as initialisation"
        from keras.models import model_from_json

        json_file = open(DATA_DIR+"/"+OUT_NAME+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(DATA_DIR+"/"+OUT_NAME+".h5")

        classification_model = loaded_model


    # 8. Compile model
    classification_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 9. Fit model on training data

    model_json = classification_model.to_json()
    with open(DATA_DIR+"/"+OUT_NAME+".json", "w") as json_file:
        json_file.write(model_json)

    best_eval_consecutive = 0
    best_eval_random = 0
    for e in range(NUM_EPOCHS):
        print e
        classification_model.fit_generator(myGenerator(), SAMPLES, epochs=1, verbose=1)


        # 10. Evaluate model on test data
        results = classification_model.evaluate(X_test_consecutive, y_test_consecutive, verbose=1)
        print "CONSECUTIVE", results

        if results[1] > best_eval_consecutive:
            best_eval_consecutive = results[1]
            print "saving weights"
            classification_model.save_weights(DATA_DIR+"/"+OUT_NAME+".h5")

        results = classification_model.evaluate(X_test_random, y_test_random, verbose=1)
        print "RANDOM", results

        if results[1] > best_eval_random:
            best_eval_random = results[1]
            print "saving weights"
            classification_model.save_weights(DATA_DIR + "/" + OUT_NAME + "_rand.h5")
