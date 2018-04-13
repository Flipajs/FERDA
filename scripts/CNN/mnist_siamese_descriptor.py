# based on https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py

'''Trains a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

# References

- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 97.2% test accuracy after 20 epochs.
2 seconds per epoch on a Titan X Maxwell GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import argparse
from scripts.CNN.train_siamese_contrastive_lost import contrastive_loss2
from scripts.CNN.prepare_siamese_data import get_region_crop
from scripts.CNN.prepare_siamese_data import ELLIPSE_DILATION, APPLY_ELLIPSE, OFFSET, MASK_SIGMA
import cv2
from keras.utils import CustomObjectScope

def relu6(x):
  return K.relu(x, max_value=6)


def normalize_and_prepare_imgs(imgs):
    imgs = np.array(imgs)
    imgs = imgs.astype('float32')
    imgs /= 255

    return imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train siamese CNN with contrastive loss')

    parser.add_argument('--datadir', type=str,
                        default='/Users/flipajs/Documents/wd/FERDA/april-paper/Cam1_clip_arena_fixed',
                        help='path to dataset')
    parser.add_argument('--epochs', type=int,
                        default=10,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=128,
                        help='batch size')
    parser.add_argument('--weights_name', type=str, default='best_weights',
                        help='name used for saving intermediate results')
    parser.add_argument('--num_negative', type=int, default=1,
                        help='name used for saving intermediate results')
    parser.add_argument('--continue_training', type=bool, default=False,
                        help='if True, use --weights as initialisation')

    args = parser.parse_args()
    from scripts.CNN.train_siamese_contrastive_lost import create_base_network10, euclidean_distance, eucl_dist_output_shape

    input_shape = (90, 90, 3)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    architecture = create_base_network10
    base_network = architecture(input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    model.load_weights(args.datadir+"/best_model_976_weights.h5")
    new_model = model.layers[2]
    # https://github.com/tensorflow/tensorflow/issues/17191
    # with CustomObjectScope({'relu6': relu6, 'contrastive_loss2': contrastive_loss2}):
        # m = load_model(args.datadir+"/best_model.h5", compile=False)
        # new_model = m.layers[2]

    from core.project.project import Project
    p = Project()
    p.load(args.datadir)
    vm = p.get_video_manager()

    imgs = []
    r_ids = []
    descriptors = {}
    batch_size = 300
    for frame in tqdm(range(p.num_frames())):
        img = vm.get_frame(frame)
        tracklets = filter(lambda x: x.is_single(), p.chm.tracklets_in_frame(frame))

        for tracklet in tracklets:
            region = tracklet.get_region_in_frame(p.gm, frame)
            crop = get_region_crop(region, img, APPLY_ELLIPSE, ELLIPSE_DILATION, MASK_SIGMA , OFFSET)
            imgs.append(crop)
            r_ids.append(region.id())

        if len(imgs) >= batch_size:
            imgs = normalize_and_prepare_imgs(imgs)
            descs = new_model.predict(imgs)

            for k, r_id in enumerate(r_ids):
                descriptors[r_id] = descs[k, :]

            imgs = []
            r_ids = []

    # Do the rest
    imgs = normalize_and_prepare_imgs(imgs)
    descs = new_model.predict(imgs)

    for k, r_id in enumerate(r_ids):
        descriptors[r_id] = descs[k, :]

    import pickle
    with open(args.datadir+'/descriptors.pkl', 'wb') as f:
        pickle.dump(descriptors, f)

    print("DONE")
