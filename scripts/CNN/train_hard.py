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
from keras import backend as K
from keras.losses import mse
from keras.callbacks import Callback
from keras.models import save_model

class NBatchLogger(Callback):
    def __init__(self,display=100):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.seen = 0
        self.display = display

    def on_batch_end(self,batch,logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            print '\n{0}/{1} - Batch Loss: {2}'.format(self.seen,self.params['samples'],
                                                logs.get('loss'))

def my_loss(y_true, y_pred):
    margin = 1.

    # we want to have vectors having norm
    # norm = K.sum(K.abs(K.sqrt(K.sum(K.square(y_pred), -1)) - 1))
    # y_pred = K.clip(y_pred, 0.01, 1.)
    regul = K.maximum(0., 1 - K.sum(y_pred, -1))
    p1 = y_pred[0::5, :]
    p2 = y_pred[1::5, :]
    n1 = y_pred[2::5, :]
    n2 = y_pred[3::5, :]
    n3 = y_pred[4::5, :]

    test = 0
    if y_true[0] != 1.0 or y_true[1] != 1.0 or y_true[2] != 0.0 or y_true[3] != 0.0 or y_true[4] != 0.0:
        test = 10000

    pos_val = K.sqrt(K.sum(K.square(p1 - p2), -1))
    neg_val1 = K.sqrt(K.sum(K.square(p1 - n1), -1))
    neg_val2 = K.sqrt(K.sum(K.square(p1 - n2), -1))
    neg_val3 = K.sqrt(K.sum(K.square(p1 - n3), -1))
    # neg_val2 = K.sqrt(K.sum(K.square(p2 - n)))
    neg_val = K.minimum(K.minimum(neg_val1, neg_val2), neg_val3)

    # reg = K.maximum(0., 1 - neg_val)
    # return margin + pos_val - neg_val
    val = K.mean(K.maximum(0., margin + pos_val - neg_val)) + regul + test
    return val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train siamese CNN with HARD loss (https://github.com/DagnyT/hardnet/)')

    parser.add_argument('--datadir', type=str,
                        default='/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_zebrafish',
                        help='path to dataset')
    parser.add_argument('--epochs', type=int,
                        default=1,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=32,
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

    # x = Conv2D(32, (3, 3))(animal_input)
    # x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    # x = MaxPooling2D((2, 2))(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(16, (3, 3))(x)
    # x = Conv2D(8, (3, 3))(x)

    x = Conv2D(32, (3, 3))(animal_input)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = Conv2D(32, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(64, (3, 3))(x)
    x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3))(x)
    # x = MaxPooling2D((2, 2))(x)
    x = Conv2D(8, (3, 3))(x)
    # out_a = vision_model(animal_a)
    x = Flatten()(x)
    # out = Dense(128, activation='relu')(out_a)
    out = Dense(6, activation='softmax')(x)
    classification_model = Model(animal_input, out)
    classification_model.load_weights('../data_cam1/cam1_softmax2.h5')
    # classification_model.load_weights('/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/cam1_softmax2.h5')
    print "weights loaded"

    # x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu', kernel_initializer='normal')(x)

    model = Model(animal_input, x)
    model.summary()
    # plot_model(classification_model, show_shapes=True, to_file='complete_model.png')

    # y_train = np.zeros((y_train.shape[0], 32))

    out_batch = NBatchLogger(display=1)

    # 8. Compile model
    # classification_model.compile(loss=my_loss, optimizer='adam')
    model.compile(loss=my_loss, optimizer='adam')


    from scipy import stats
    for e in range(args.epochs):
        print y_train
        model.fit(X_train, y_train, batch_size=(2 + args.num_negative) * args.batch_size, epochs=1, callbacks=[out_batch])

        pred = model.predict(X_test)

        np.set_printoptions(precision=2)
        print pred
        print stats.describe(pred)


    model.save('my_model.h5')

    with h5py.File(args.datadir+'/pred.h5', 'w') as hf:
        hf.create_dataset("data", data=pred)
