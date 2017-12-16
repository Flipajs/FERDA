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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train siamese CNN with HARD loss (https://github.com/DagnyT/hardnet/)')
    # Model options

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
    parser.add_argument('--continue_training', type=bool, default=False,
                        help='if True, use --weights as initialisation')

    args = parser.parse_args()


    with h5py.File(args.datadir + '/imgs_a_train.h5', 'r') as hf:
        X_train_a = hf['data'][:]

    with h5py.File(args.datadir + '/imgs_a_test.h5', 'r') as hf:
        X_test_a = hf['data'][:]

    with h5py.File(args.datadir + '/imgs_b_train.h5', 'r') as hf:
        X_train_b = hf['data'][:]

    with h5py.File(args.datadir + '/imgs_b_test.h5', 'r') as hf:
        X_test_b = hf['data'][:]

    with h5py.File(args.datadir + '/labels_train.h5', 'r') as hf:
        y_train = hf['data'][:]

    with h5py.File(args.datadir + '/labels_test.h5', 'r') as hf:
        y_test = hf['data'][:]


    print "train shape", X_train_a.shape
    print "test shape", X_test_a.shape

    # IMPORTANT!!
    X_train_a = X_train_a.astype('float32')
    X_train_b = X_train_b.astype('float32')
    X_test_a = X_test_a.astype('float32')
    X_test_b = X_test_b.astype('float32')
    X_train_a /= 255
    X_train_b /= 255
    X_test_a /= 255
    X_test_b /= 255

    np.random.seed(123)  # for reproducibility

    im_dim = 3
    im_h = 90
    im_w = 90

    # First, define the vision modules
    animal_input = Input(shape=X_train_a.shape[1:])

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

    vision_model = Model(animal_input, x)

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
    classification_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 9. Fit model on training data

    # checkpointer = ModelCheckpoint(filepath='weights.h5', verbose=1, save_best_only=True)
    checkpoint = ModelCheckpoint(args.weights_name+'.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    classification_model.fit([X_train_a, X_train_b], y_train, validation_split=0.05,
              batch_size=args.batch_size, epochs=args.epochs, verbose=1, callbacks=callbacks_list)


    model_json = classification_model.to_json()
    with open("model_"+args.weights_name+".json", "w") as json_file:
        json_file.write(model_json)

    classification_model.save_weights(args.weights_name+".h5")

    model_json = vision_model.to_json()
    with open("vision_model_"+args.weights_name+".json", "w") as json_file:
        json_file.write(model_json)

    vision_model.save_weights("vision_"+args.weights_name+".h5")

    # 10. Evaluate model on test data
    results = classification_model.evaluate([X_test_a, X_test_b], y_test, verbose=1)
    print results
