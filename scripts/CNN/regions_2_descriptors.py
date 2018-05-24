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

import h5py
import argparse


def process_batch(imgs_batch, ids_batch):
    imgs_batch = np.array(imgs_batch)
    imgs_batch = imgs_batch.astype('float32')
    imgs_batch /= 255

    descs = descriptor_model.predict(imgs_batch)

    for k, r_id in enumerate(ids_batch):
        descriptors[r_id] = descs[k, :]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train siamese CNN with contrastive loss')

    parser.add_argument('--datadir', type=str,
                        default='/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/',
                        help='path to dataset')
    parser.add_argument('--weights_name', type=str, default='best_weights',
                        help='name used for saving intermediate results')
    parser.add_argument('--num_negative', type=int, default=1,
                        help='name used for saving intermediate results')
    parser.add_argument('--continue_training', type=bool, default=False,
                        help='if True, use --weights as initialisation')

    args = parser.parse_args()

    m = load_model(args.datadir+"/best_model_on6_300_ft.h5", compile=False)
    descriptor_model = m.layers[2]

    P_WD = '/Users/flipajs/Documents/wd/FERDA/april-paper/Cam1_clip'
    # P_WD = '/Users/flipajs/Documents/wd/FERDA/zebrafish_new'
    from core.project.project import Project
    p = Project()
    p.load(P_WD)

    # go in frame order (optimized video accesss) and pick regions from single-ID tracklets. Compute for them descriptor
    imgs_batch = []
    ids_batch = []
    descriptors = {}

    from utils.img import get_safe_selection
    from utils.img import apply_ellipse_mask
    import cv2
    # TODO: project parameter?
    OFFSET = 45
    ELLIPSE_DILATION = 10
    MASK_SIGMA = 10
    BATCH_SIZE = 500
    APPLY_ELLIPSE = True

    np.set_printoptions(precision=2)
    from tqdm import tqdm

    vm = p.get_video_manager()
    last = None
    for frame in tqdm(range(vm.total_frame_count())):
        img = None
        for tracklet in p.chm.tracklets_in_frame(frame):
            if tracklet.is_single():
                region = tracklet.get_region_in_frame(p.gm, frame)

                # optimization for frames where are 0 single-ID tracklets
                if img is None:
                    img = vm.get_frame(region.frame())

                y, x = region.centroid()
                crop = get_safe_selection(img, y - OFFSET, x - OFFSET, 2 * OFFSET, 2 * OFFSET)
                if APPLY_ELLIPSE:
                    crop = apply_ellipse_mask(region, crop, MASK_SIGMA, ELLIPSE_DILATION)

                imgs_batch.append(crop)
                ids_batch.append(region.id())

        if len(imgs_batch) > BATCH_SIZE:
            process_batch(imgs_batch, ids_batch)
            imgs_batch, ids_batch = [], []

    # Do the rest...
    if len(imgs_batch):
        process_batch(imgs_batch, ids_batch)

    import pickle
    with open(p.working_directory+'/descriptors.pkl', 'wb') as f:
        pickle.dump(descriptors, f)

    print("DONE")
