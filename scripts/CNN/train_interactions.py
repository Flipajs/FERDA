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
from keras.layers import Dense, BatchNormalization, Activation, AveragePooling2D
from keras import backend as K
from keras import layers
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




# https://github.com/nicolov/segmentation_keras/blob/master/model.py
from keras.layers import Activation, Reshape, Dropout
from keras.layers import AtrousConvolution2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential


#
# The VGG16 keras model is taken from here:
# https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# The (caffe) structure of DilatedNet is here:
# https://github.com/fyu/dilation/blob/master/models/dilation8_pascal_voc_deploy.prototxt

def get_frontend(input_width, input_height):
    model = Sequential()
    # model.add(ZeroPadding2D((1, 1), input_shape=(input_width, input_height, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', input_shape=(input_width, input_height, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))

    # Compared to the original VGG16, we skip the next 2 MaxPool layers,
    # and go ahead with dilated convolutional layers instead

    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1'))
    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2'))
    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3'))

    # Compared to the VGG16, we replace the FC layer with a convolution

    model.add(AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, 1, 1, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    # Note: this layer has linear activations, not ReLU
    model.add(Convolution2D(21, 1, 1, activation='linear', name='fc-final'))

    # model.layers[-1].output_shape == (None, 16, 16, 21)
    return model


def add_softmax(model):
    """ Append the softmax layers to the frontend or frontend + context net. """
    # The softmax layer doesn't work on the (width, height, channel)
    # shape, so we reshape to (width*height, channel) first.
    # https://github.com/fchollet/keras/issues/1169
    _, curr_width, curr_height, curr_channels = model.layers[-1].output_shape

    model.add(Reshape((curr_width * curr_height, curr_channels)))
    model.add(Activation('softmax'))
    # Technically, we need another Reshape here to reshape to 2d, but TF
    # the complains when batch_size > 1. We're just going to reshape in numpy.
    # model.add(Reshape((curr_width, curr_height, curr_channels)))

    return model


def add_context(model):
    """ Append the context layers to the frontend. """
    model.add(ZeroPadding2D(padding=(33, 33)))
    model.add(Convolution2D(42, 3, 3, activation='relu', name='ct_conv1_1'))
    model.add(Convolution2D(42, 3, 3, activation='relu', name='ct_conv1_2'))
    model.add(AtrousConvolution2D(84, 3, 3, atrous_rate=(2, 2), activation='relu', name='ct_conv2_1'))
    model.add(AtrousConvolution2D(168, 3, 3, atrous_rate=(4, 4), activation='relu', name='ct_conv3_1'))
    model.add(AtrousConvolution2D(336, 3, 3, atrous_rate=(8, 8), activation='relu', name='ct_conv4_1'))
    model.add(AtrousConvolution2D(672, 3, 3, atrous_rate=(16, 16), activation='relu', name='ct_conv5_1'))
    model.add(Convolution2D(672, 3, 3, activation='relu', name='ct_fc1'))
    model.add(Convolution2D(21, 1, 1, name='ct_final'))


    return model











def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def model():
    global NUM_PARAMS, DATA_DIR, CONTINUE

    img_input = Input(shape=(200, 200, 3))

    # build the VGG16 network with our input_img as input
    first_layer = Conv2D(64, (3, 3), input_shape=img_input, padding='same')


    x = Conv2D(64, (3, 3), padding='same', activation='relu')(img_input)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64 (3, 3), dilation_rate=(2, 2), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), dilation_rate=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    x = Flatten()(x)
    x = Dense(NUM_PARAMS, activation='linear')(x)
    vision_model = Model(img_input, x)
    vision_model.summary()

    # model = Sequential()
    # model.add(first_layer)
    # model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #
    # model.add(Dense(NUM_PARAMS, activation='linear'))

    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    return model

# def model():
#     global NUM_PARAMS, DATA_DIR, CONTINUE
#
#     img_input = Input(shape=(200, 200, 3))
#
#     if K.image_data_format() == 'channels_last':
#         bn_axis = 3
#     else:
#         bn_axis = 1
#
#     x = Conv2D(
#         64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
#     x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
#     x = Activation('relu')(x)
#     x = MaxPooling2D((3, 3), strides=(2, 2))(x)
#
#     x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
#
#     x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
#
#     # x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
#     # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
#     # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
#     # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
#     # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
#     # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
#     #
#     # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
#     # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#     # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
#
#     x = AveragePooling2D((7, 7), name='avg_pool')(x)
#
#     x = Flatten()(x)
#     out = Dense(NUM_PARAMS, activation='linear', name='fc1000')(x)
#
#     model = Model(img_input, out, name='resnet50')
#     # model = Model(input_shape, out)
#
#     if CONTINUE:
#         model.load_weights(DATA_DIR + "/interaction_weights_"+str(NUM_PARAMS)+".h5")
#
#     model.summary()
#     model.compile(loss='mean_squared_error',
#                   optimizer='adam')
#
#     return model



# def model():
#     global NUM_PARAMS, DATA_DIR, CONTINUE
#     input_shape = Input(shape=(200, 200, 3))
#
#     # LOAD...
#     from keras.models import model_from_json
#
#     json_file = open(ROOT_DIR+'/vision_model_'+WEIGHTS+'.json', 'r')
#     vision_model_json = json_file.read()
#     json_file.close()
#     vision_model = model_from_json(vision_model_json)
#     # load weights into new model
#     vision_model.load_weights(ROOT_DIR+"/vision_"+WEIGHTS+".h5")
#     # vision_model.layers.pop()
#     # vision_model.layers.pop()
#
#     vision_model.summary()
#
#     # The vision model will be shared, weights and all
#     out_a = vision_model(input_shape)
#     # out_a = Flatten()(out_a)
#     #
#     # out_a = Dense(256, activation='relu')(out_a)
#     # out_a = Dense(128, activation='relu')(out_a)
#     # out_a = Dense(32, activation='relu')(out_a)
#     out_a = Dense(64, activation='relu')(out_a)
#     out_a = Dense(32, activation='relu')(out_a)
#
#     # out = Dense(128, activation='relu')(out_a)
#     # out = Dense(K, activation='softmax')(out_a)
#     previous = NUM_PARAMS - 2
#     if NUM_PARAMS == 4 or CONTINUE:
#         previous = NUM_PARAMS
#
#     out = Dense(previous, kernel_initializer='normal', activation='linear')(out_a)
#
#     model = Model(input_shape, out)
#
#     model.load_weights(DATA_DIR + "/interaction_weights_"+str(previous)+".h5")
#     # model.load_weights(DATA_DIR + "/interaction_weights.h5")
#
#     out = Dense(NUM_PARAMS, kernel_initializer='normal', activation='linear')(out_a)
#     model = Model(input_shape, out)
#     #
#     # model =
#
#     model.summary()
#     # 8. Compile model
#     # adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#     model.compile(loss='mean_squared_error',
#                   optimizer='adam')
#
#     # model.lr.set_value(0.05)
#
#     return model


if __name__ == '__main__':
    NUM_EPOCHS = 5
    USE_PREVIOUS_AS_INIT = 0
    # K = 6
    WEIGHTS = 'best_weights'
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
        NUM_PARAMS = string.atoi(sys.argv[5])
    if len(sys.argv) > 6:
        CONTINUE = bool(string.atoi(sys.argv[6]))

    with h5py.File(DATA_DIR + '/imgs_inter_train.h5', 'r') as hf:
        X_train = hf['data'][:]

    with h5py.File(DATA_DIR + '/imgs_inter_test.h5', 'r') as hf:
        X_test = hf['data'][:]

    with h5py.File(DATA_DIR + '/results_inter_train.h5', 'r') as hf:
        y_train = hf['data'][:]

    with h5py.File(DATA_DIR + '/results_inter_test.h5', 'r') as hf:
        y_test = hf['data'][:]

    if NUM_PARAMS == 4:
        ids = np.array([0, 1, 5, 6])
    if NUM_PARAMS == 6:
        ids = np.array([0, 1, 2, 5, 6, 7])
    if NUM_PARAMS == 8:
        ids = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    if NUM_PARAMS == 10:
        ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    y_test = y_test[:, ids]
    y_train = y_train[:, ids]

    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    scaler.fit(y_train)
    # NUM_PARAMS = y_train.shape[1]
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)


    print "NUM params: ", NUM_PARAMS
    m = model()
    m.fit(X_train, y_train, validation_split=0.05, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    pred = m.predict(X_test)

    pred = scaler.inverse_transform(pred)
    y_test = scaler.inverse_transform(y_test)
    # print pred2.shape

    with h5py.File(DATA_DIR+'/predictions.h5', 'w') as hf:
        hf.create_dataset("data", data=pred)

    m.save_weights(DATA_DIR + "/interaction_weights_"+str(NUM_PARAMS)+".h5")

    model_json = m.to_json()
    with open(DATA_DIR + "/interaction_model_"+str(NUM_PARAMS)+".json", "w") as json_file:
        json_file.write(model_json)

    from sklearn.metrics import mean_squared_error
    print "MSE", mean_squared_error(y_test, pred)
    from sklearn.metrics import mean_absolute_error
    print "MAE", mean_absolute_error(y_test, pred)

    # print m.score(X_test, y_test)