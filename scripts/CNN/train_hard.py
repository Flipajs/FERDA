import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import keras
import sys
import string
import numpy as np
from keras.utils import plot_model
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, BatchNormalization, Activation
from keras.activations import sigmoid
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from scipy import misc
from keras import backend as K
from keras.losses import mse
from keras.callbacks import Callback
from keras.models import save_model
from keras.regularizers import l2

OUT_DIM = 32

def Conv2DReluBatchNorm(n_filter, w_filter, h_filter, inputs):
    return BatchNormalization()(Activation(activation='relu')(Conv2D(n_filter, (w_filter, h_filter))(inputs)))

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

def my_loss2(y_true, y_pred):
    # y_pred = K.l2_normalize(y_pred, axis=-1)
    margin = 1.
    embeddings = K.reshape(y_pred, (-1, 3, OUT_DIM))

    positive_distance = K.mean(K.square(embeddings[:,0] - embeddings[:,1]),axis=-1)
    negative_distance = K.mean(K.square(embeddings[:,0] - embeddings[:,2]),axis=-1)
    return K.mean(K.maximum(0.0, positive_distance - negative_distance + margin))

def my_loss(y_true, y_pred):
    margin = 0.1

    # we want to have vectors having norm

    # penalize_zero = K.mean(K.switch(K.less_equal(norm, 0.2), K.ones_like(norm) * 100000.0, K.zeros_like(norm)))
    # y_pred = K.clip(y_pred, 1e-14, 10.0)
    # y_pred = K.l2_normalize(y_pred, axis=-1)
    # norm = K.mean(K.abs(K.sqrt(K.sum(K.square(y_pred), -1))))
    # regul = K.maximum(0., 1 - K.sum(y_pred, -1))
    # return K.sum(y_pred, -1)
    # regul = K.mean(K.maximum(0., 1 - K.sum(y_pred, -1)))
    p1 = y_pred[0::3, :]
    p2 = y_pred[1::3, :]
    n1 = y_pred[2::3, :]
    # n2 = y_pred[3::5, :]
    # n3 = y_pred[4::5, :]

    eps = 1e-16
    pos_val = K.sqrt(K.sum(K.square(p1 - p2) + eps, -1))
    neg_val = K.sqrt(K.sum(K.square(p1 - n1) + eps, -1))
    # neg_val2 = K.sqrt(K.sum(K.square(p1 - n2), -1))
    # neg_val3 = K.sqrt(K.sum(K.square(p1 - n3), -1))
    # neg_val2 = K.sqrt(K.sum(K.square(p2 - n)))
    # neg_val = K.minimum(K.minimum(neg_val1, neg_val2), neg_val3)
    # reg = K.maximum(0., 1 - neg_val)
    # results = K.mean(K.maximum(0., margin + pos_val - neg_val)) + norm
    # results = K.repeat_elements(results, 3, -1)
    # results = K.switch(K.less_equal(test, 0.), K.zeros_like(test), results)
    # results = K.mean(K.maximum(0., margin + pos_val - neg_val))+norm
    # results = K.mean(K.maximum(0., pos_val))

    # exp_pos = K.exp(2.0 - pos_val)
    # exp_den = exp_pos + K.exp(2.0 - neg_val) + 1e-16
    # results = - K.log(exp_pos / exp_den)


    # loss = K.clip(margin - neg_val, 0.0, margin) + K.clip(pos_val, 0, margin)
    # loss = K.clip(margin - neg_val, 0.0, np.inf) + pos_val
    # loss = K.clip(pos_val - neg_val, 0.0, np.inf)

    # loss = K.switch(K.less(pos_val, neg_val), K.zeros_like(pos_val), K.ones_like(pos_val))
    # return K.less(neg_val, pos_val)
    return pos_val-neg_val
    # loss = K.cast(K.less(neg_val, pos_val), dtype=np.float32)
    # loss = neg_val
    # return loss
    # return K.mean(neg_val)
    return K.repeat_elements(loss, 3, -1)

    # return K.mean(loss)

    # return K.mean(mar-neg_val)+penalize_zero
    # results = K.repeat_elements(results), 5, -1)
    # results = K.switch(K.less_equal(y_true[:, 0], 0.), K.zeros_like(y_true[:, 0]), results)


    results = results

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train siamese CNN with HARD loss (https://github.com/DagnyT/hardnet/)')

    parser.add_argument('--datadir', type=str,
                        default='/Users/flipajs/Documents/wd/FERDA/CNN_hard_datagen',
                        help='path to dataset')
    parser.add_argument('--epochs', type=int,
                        default=20,
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


    print "train shape {}, min: {} max: {}".format(X_train.shape, X_train.min(), X_train.max())
    print "test shape {}, min: {} max: {}".format(X_test.shape, X_test.min(), X_test.max())
    print "y_train shape {}, min: {} max: {}".format(y_train.shape, y_train.min(), y_train.max())

    np.random.seed(123)  # for reproducibility

    im_dim = 3
    im_h = 32
    im_w = 32

    # First, define the vision modules
    animal_input = Input(shape=X_train.shape[1:])

    # x = Conv2DReluBatchNorm(32, 3, 3, animal_input)
    x = Conv2D(8, (3, 3), activation='relu')(animal_input)
    x = Conv2D(8, (3, 3), activation='relu', dilation_rate=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    # x = Conv2D(32, (3, 3), dilation_rate=(2, 2))(x)
    # x = Conv2DReluBatchNorm(32, 3, 3, x)
    # x = MaxPooling2D((2, 2))(x)
    # x = Conv2DReluBatchNorm(16, 3, 3, x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    # x = Conv2DReluBatchNorm(8, 3, 3, x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = Flatten()(x)

    # out = Dense(6, activation='softmax')(x)
    # classification_model = Model(animal_input, out)
    # classification_model.load_weights('../data_cam1/cam1_softmax2.h5')
    # classification_model.load_weights('/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/cam1_softmax2.h5')
    print "weights loaded"

    # x = Flatten()(x)
    # x = Dense(32, activation='sigmoid')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(OUT_DIM, activation='sigmoid', kernel_initializer='random_uniform')(x)
    # x = BatchNormalization()(x)
    # out = Activation('sigmoid')(x)
    # x = Dense(8, activation='sigmoid', kernel_initializer='uniform')(x)

    model = Model(animal_input, x)
    model.summary()
    # plot_model(classification_model, show_shapes=True, to_file='complete_model.png')

    # y_train = np.zeros((y_train.shape[0], 32))

    out_batch = NBatchLogger(display=1)

    # 8. Compile model
    # classification_model.compile(loss=my_loss, optimizer='adam')
    from keras.optimizers import Adam
    # optimizer = Adam(lr=0.01)
    model.compile(loss=my_loss2, optimizer='adam')


    from scipy import stats
    for e in range(args.epochs):
        model.fit(X_train, y_train, batch_size=(2 + args.num_negative) * args.batch_size, epochs=1, callbacks=[out_batch])

        pred = model.predict(X_test)

        np.set_printoptions(precision=2)
        print pred

        pos_d = []
        neg_d = []
        for i in range(pred.shape[0] / (2 + args.num_negative)):
            p1 = pred[i * (2 + args.num_negative)]
            p2 = pred[i * (2 + args.num_negative) + 1]
            n = pred[i * (2 + args.num_negative) + 2]

            pos_d.append(np.linalg.norm(p1 - p2))
            neg_d.append(min(np.linalg.norm(n - p1), np.linalg.norm(n - p2)))

        from scipy import stats

        # print stats.describe(pos_d)
        # print stats.describe(neg_d)

        corr = np.sum(np.array(pos_d) < np.array(neg_d))
        print "{}, {:.2%}".format(corr, corr / float(len(pos_d)))


    model.save('my_model.h5')

    with h5py.File(args.datadir+'/pred.h5', 'w') as hf:
        hf.create_dataset("data", data=pred)
