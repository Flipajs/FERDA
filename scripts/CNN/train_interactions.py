import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import sys
import string
import numpy as np
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


ROOT_DIR = '/home/threedoid/cnn_descriptor/'
# ROOT_DIR = '/Users/flipajs/Documents/wd/FERDA/cnn_exp'
DATA_DIR = ROOT_DIR + '/data'
BATCH_SIZE = 32
TWO_TESTS = True

WEIGHTS = 'cam3_zebr_weights_vgg'
NUM_PARAMS = 6

def model():
    input_shape = Input(shape=(200, 200, 3))

    # LOAD...
    from keras.models import model_from_json

    json_file = open(ROOT_DIR+'/vision_model_'+WEIGHTS+'.json', 'r')
    vision_model_json = json_file.read()
    json_file.close()
    vision_model = model_from_json(vision_model_json)
    # load weights into new model
    vision_model.load_weights(ROOT_DIR+"/vision_"+WEIGHTS+".h5")
    vision_model.layers.pop()
    vision_model.layers.pop()

    vision_model.summary()

    # The vision model will be shared, weights and all
    out_a = vision_model(input_shape)
    # out_a = Flatten()(out_a)
    #
    # out_a = Dense(256, activation='relu')(out_a)
    # out_a = Dense(128, activation='relu')(out_a)
    # out_a = Dense(32, activation='relu')(out_a)
    out_a = Dense(64, activation='relu')(out_a)
    out_a = Dense(32, activation='relu')(out_a)

    # out = Dense(128, activation='relu')(out_a)
    # out = Dense(K, activation='softmax')(out_a)
    out = Dense(NUM_PARAMS, kernel_initializer='normal', activation='linear')(out_a)
    model = Model(input_shape, out)
    model.summary()
    # 8. Compile model
    # adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    # model.lr.set_value(0.05)

    return model


if __name__ == '__main__':
    NUM_EPOCHS = 5
    USE_PREVIOUS_AS_INIT = 0
    K = 6
    WEIGHTS = 'best_weights'
    OUT_NAME = 'softmax'
    CONTINUE = False
    SAMPLES = 2000

    if len(sys.argv) > 1:
        DATA_DIR = ROOT_DIR + '/' + sys.argv[1]
    if len(sys.argv) > 2:
        NUM_EPOCHS = string.atoi(sys.argv[2])
    if len(sys.argv) > 3:
        BATCH_SIZE = string.atoi(sys.argv[3])
    if len(sys.argv) > 4:
        WEIGHTS = sys.argv[4]
    if len(sys.argv) > 5:
        OUT_NAME = sys.argv[5]

    with h5py.File(DATA_DIR + '/imgs_inter_train.h5', 'r') as hf:
        X_train = hf['data'][:]

    with h5py.File(DATA_DIR + '/imgs_inter_test.h5', 'r') as hf:
        X_test = hf['data'][:]

    with h5py.File(DATA_DIR + '/results_inter_train.h5', 'r') as hf:
        y_train = hf['data'][:]

    with h5py.File(DATA_DIR + '/results_inter_test.h5', 'r') as hf:
        y_test = hf['data'][:]

    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    scaler.fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_train)

    m = model()
    m.fit(X_train, y_train, validation_split=0.05, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # evaluate model with standardized dataset
    # estimator = KerasRegressor(build_fn=model, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    #
    # kfold = KFold(n_splits=10, random_state=seed)
    # results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    # estimator.fit(X_train, y_train, validation_split=0.05)

    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import StandardScaler
    #
    # estimators = []
    # estimators.append(('standardize', StandardScaler()))
    # estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2)))
    # pipeline = Pipeline(estimators)
    # # X_train, X_test, y_train, y_test = train_test_split(X, Y,
    # #                                                     train_size=0.75, test_size=0.25)
    # pipeline.fit(X_train, y_train, mlp__validation_split=0.05)

    pred = m.predict(X_test)

    pred = scaler.inverse_transform(pred)

    with h5py.File(DATA_DIR+'/predictions.h5', 'w') as hf:
        hf.create_dataset("data", data=pred)

    from sklearn.metrics import mean_squared_error
    print "MSE", mean_squared_error(y_test, pred)
    from sklearn.metrics import mean_absolute_error
    print "MAE", mean_absolute_error(y_test, pred)

    print m.score(X_test, y_test)

    m.save_weights(DATA_DIR + "/weights.h5")

    model_json = m.to_json()
    with open(DATA_DIR + "/model.json", "w") as json_file:
        json_file.write(model_json)
