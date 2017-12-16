import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import keras
import sys
import string
import numpy as np
from keras.utils import plot_model
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from scipy import misc

#
#
# def myGenerator():
#     global DATA_DIR, BATCH_SIZE
#     with h5py.File(DATA_DIR + '/imgs_multi_train.h5', 'r') as hf:
#         X_train = hf['data'][:]
#
#     with h5py.File(DATA_DIR + '/labels_multi_train.h5', 'r') as hf:
#         y_train = hf['data'][:]
#
#     X_train = X_train.astype('float32')
#     X_train /= 255
#
#     N_NEGATIVE = 1
#
#     positive = np.argwhere(y_train == 1)
#     positive.shape = (positive.shape[0], )
#     negative = np.argwhere(y_train == 0)
#     negative.shape = (negative.shape[0], )
#
#     while 1:
#         pos = np.random.choice(positive, 2)
#         neg = np.random.choice(negative, N_NEGATIVE)
#
#         ids = np.hstack([pos, neg])
#         yield X_train[ids, :], y_train[pos, :]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train siamese CNN with HARD loss (https://github.com/DagnyT/hardnet/)')

    parser.add_argument('--datadir', type=str,
                        default='/home/threedoid/cnn_descriptor/data',
                        help='path to dataset')
    parser.add_argument('--epochs', type=int,
                        default=5,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--weights_name', type=str, default='best_weights',
                        help='name used for saving intermediate results')
    parser.add_argument('--num_negative', type=int, default=1,
                        help='name used for saving intermediate results')
    parser.add_argument('--continue_training', type=bool, default=False,
                        help='if True, use --weights as initialisation')

    args = parser.parse_args()


    with h5py.File(args.datadir + '/imgs_train_hard_'+str(args.num_negative)+'.h5', 'r') as hf:
        X_train = hf['data'][:]

    with h5py.File(args.datadir + '/imgs_test_hard_'+str(args.num_negative)+'.h5', 'r') as hf:
        X_test = hf['data'][:]

    with h5py.File(args.datadir + '/labels_train_hard_'+str(args.num_negative)+'.h5', 'r') as hf:
        y_train = hf['data'][:]

    with h5py.File(args.datadir + '/labels_test_hard_'+str(args.num_negative)+'.h5', 'r') as hf:
        y_test = hf['data'][:]


    print "train shape", X_train.shape
    print "test shape", X_test.shape

    np.random.seed(123)  # for reproducibility

    im_dim = 3
    im_h = 90
    im_w = 90

    # First, define the vision modules
    animal_input = Input(shape=X_train.shape[1:])

    x = Conv2D(32, (3, 3))(animal_input)
    x = Conv2D(32, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = Conv2D(32, (3, 3))(x)
    x = Conv2D(32, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3))(x)
    x = Conv2D(64, (3, 3))(x)
    x = Conv2D(32, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3))(x)
    x = Conv2D(8, (3, 3))(x)

    x = Flatten()(x)

    classification_model = Model(animal_input, x)

    plot_model(classification_model, show_shapes=True, to_file='complete_model.png')

    ########### load weights... ugly way how to do it now...
    if args.continue_training:
        print "Using last saved weights as initialisation"
        from keras.models import model_from_json
        json_file = open('model_'+args.weights_name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(args.weights_name+".h5")

        classification_model = loaded_model

    # 8. Compile model
    classification_model.compile(loss=my_loss,
                  optimizer='adam')

