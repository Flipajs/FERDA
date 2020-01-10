# compute re-identification descriptors for all regions in single tracklets
import numpy as np
import pickle
from os.path import join
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input
from tqdm import tqdm
import argparse
import logging
from core.reidentification.prepare_siamese_data import DEFAULT_PARAMETERS, get_region_crop
from core.reidentification.train_siamese_contrastive_lost import create_base_network10, euclidean_distance, \
    eucl_dist_output_shape
from core.project.project import Project

logger = logging.getLogger(__name__)


def normalize_and_prepare_imgs(imgs):
    imgs = np.array(imgs)
    imgs = imgs.astype('float32')
    imgs /= 255
    return imgs


def compute_descriptors(project_dir, model_weights_path, add_missing=False, parameters=None):
    logger.info('computing re-id descriptors')
    if parameters is None:
        parameters = DEFAULT_PARAMETERS
    model = create_model(model_weights_path)

    p = Project(project_dir)
    vm = p.get_video_manager()
    descriptors = {}
    if add_missing:
        try:
            with open(join(project_dir, 'descriptors.pkl'), 'rb') as f:
                descriptors = pickle.load(f)
        except:
            pass
    imgs = []
    r_ids = []
    batch_size = 300
    for frame in tqdm(list(range(p.num_frames())), desc='computing re-identification descriptors'):
        img = vm.get_frame(frame)
        tracklets = [x for x in p.chm.tracklets_in_frame(frame) if x.is_single()]

        for tracklet in tracklets:
            region = tracklet.get_region_in_frame(frame)
            if region.id() in descriptors:
                continue

            crop = get_region_crop(region, img, **parameters)
            imgs.append(crop)
            r_ids.append(region.id())

        if len(imgs) >= batch_size:
            imgs = normalize_and_prepare_imgs(imgs)
            descs = model.predict(imgs)

            for k, r_id in enumerate(r_ids):
                descriptors[r_id] = descs[k, :]

            imgs = []
            r_ids = []

    # Do the rest
    if imgs:
        imgs = normalize_and_prepare_imgs(imgs)
        descs = model.predict(imgs)
        for k, r_id in enumerate(r_ids):
            descriptors[r_id] = descs[k, :]
    with open(join(project_dir, 'descriptors.pkl'), 'wb') as f:
        pickle.dump(descriptors, f)
    logger.info('done')


def create_model(model_weights_path):
    input_shape = (90, 90, 3)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    architecture = create_base_network10
    base_network = architecture(input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model_ = Model([input_a, input_b], distance)
    model_.load_weights(model_weights_path)
    model = model_.layers[2]
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute re-identification descriptors for tracklets')
    parser.add_argument('--weights', type=str, help='filename of the reidentification model weights')
    parser.add_argument('--project-dir', type=str, help='project directory')
    parser.add_argument('--add-missing', default=False, action='store_true',
                        help='if used - only ids missing in descriptors.pkl will be computed')

    args = parser.parse_args()
    compute_descriptors(args.project_dir, args.weights, args.add_missing)
