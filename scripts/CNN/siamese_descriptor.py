from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import pickle
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

def normalize_and_prepare_imgs(imgs):
    imgs = np.array(imgs)
    imgs = imgs.astype('float32')
    imgs /= 255

    return imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train siamese CNN with contrastive loss')

    parser.add_argument('--datadir', type=str,
                        # default='/Users/flipajs/Documents/wd/FERDA/april-paper/Cam1_clip_arena_fixed',
                        default='/Users/flipajs/Documents/wd/FERDA/april-paper/Sowbug3-fixed-segmentation',
                        # default='/Users/flipajs/Documents/wd/FERDA/april-paper/Camera3-5min',
                        # default='/Users/flipajs/Documents/wd/FERDA/april-paper/Camera3-5min',
                        # default='/Users/flipajs/Documents/wd/FERDA/april-paper/5Zebrafish_nocover_22min',
                        help='path to dataset')

    parser.add_argument('--add_missing', default=False, action='store_true',
                        help='if used - only ids missing in descriptors.pkl will be computed')

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
    model.load_weights(args.datadir+"/best_model_998_weights.h5")
    # model.load_weights(args.datadir+"/best_model_996_weights.h5")
    # model.load_weights(args.datadir+"/best_model_980_weights.h5")
    # model.load_weights(args.datadir+"/best_model_967_weights.h5")
    # Cam1
    # model.load_weights(args.datadir+"/best_model_996_weights.h5")
    new_model = model.layers[2]

    from core.project.project import Project
    p = Project()
    p.load(args.datadir)
    vm = p.get_video_manager()

    descriptors = {}
    if args.add_missing:
        try:
            with open(args.datadir + '/descriptors.pkl', 'rb') as f:
                descriptors = pickle.load(f)
        except:
            pass


    imgs = []
    r_ids = []
    batch_size = 100
    for frame in tqdm(range(p.num_frames())):
        img = vm.get_frame(frame)
        tracklets = filter(lambda x: x.is_single(), p.chm.tracklets_in_frame(frame))

        for tracklet in tracklets:
            region = tracklet.get_region_in_frame(p.gm, frame)
            if region.id() in descriptors:
                continue

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

    with open(args.datadir+'/descriptors.pkl', 'wb') as f:
        pickle.dump(descriptors, f)

    print("DONE")
