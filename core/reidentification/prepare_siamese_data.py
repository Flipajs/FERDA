# prepare datasets for training a dimensionality reduction network for re-identification
from __future__ import print_function
from __future__ import print_function
import numpy as np
import tqdm
from cachetools import LRUCache
from tqdm import tqdm
import random
import h5py
from os.path import join, exists
from core.project.project import Project
from utils.img import get_safe_selection
from utils.img import apply_ellipse_mask
import yaml

DEFAULT_PARAMETERS = {
    'offset': 45,
    'ellipse_dilation': 10,
    'mask_sigma': 10,
    'apply_ellipse': True,
    'align': True,
    'num_examples': 5000,
    'test_ratio': .1,
}

cache = None


def get_region_image(region, **kwargs):
    global cache
    img = cache[region.frame()]
    # img = vm.get_frame(region.frame())
    crop = get_region_crop(region, img, **kwargs)
    return crop


def get_region_crop(region, img, apply_ellipse=True, ellipse_dilation=10, mask_sigma=10, offset=45, align=True, **kwargs):
    y, x = region.centroid()
    crop = get_safe_selection(img, y - offset, x - offset, 2 * offset, 2 * offset)
    if apply_ellipse:
        crop = apply_ellipse_mask(region, crop, mask_sigma, ellipse_dilation)

    if align:
        flip = 180*random.randint(0, 1)
        import imutils
        angle = int(region.theta_ * 57.295 + flip)
        crop = imutils.rotate(crop, -angle)

    return crop


def exist_reidentification_training_data(data_dir):
    return \
        exists(join(data_dir, 'descriptor_cnn_imgs_train.h5')) and \
        exists(join(data_dir, 'descriptor_cnn_imgs_test.h5')) and \
        exists(join(data_dir, 'descriptor_cnn_labels_train.h5')) and \
        exists(join(data_dir, 'descriptor_cnn_labels_test.h5')) and \
        exists(join(data_dir, 'descriptor_cnn_params.yaml'))


def generate_reidentification_training_data(project, out_dir, parameters=None):
    if parameters is None:
        parameters = DEFAULT_PARAMETERS
    LINEAR = False
    np.set_printoptions(precision=2)

    vm = project.get_video_manager()
    global cache
    cache = LRUCache(maxsize=5000, missing=lambda x: vm.get_frame(x))

    if LINEAR:
        for i in tqdm(range(project.num_frames())):
            cache[i]

    imgs = []
    labels = []
    i = 0
    with tqdm(total=parameters['num_examples']) as pbar:
        while i < parameters['num_examples']:
            frame = random.randint(0, vm.total_frame_count())

            tracklets = filter(lambda x: x.is_single(), project.chm.tracklets_in_frame(frame))
            if len(tracklets) > 1:
                trackletA, trackletB = random.sample(tracklets, 2)
                regionA1 = trackletA.get_random_region()
                regionA2 = trackletA.get_random_region()
                regionB = trackletB.get_random_region()

                cropA1 = get_region_image(regionA1, **parameters)
                cropA2 = get_region_image(regionA2, **parameters)
                cropB = get_region_image(regionB, **parameters)

                imgs += [[cropA1, cropA2]]
                imgs += [[cropA1, cropB]]
                labels += [1, 0]

                i += 1
                pbar.update(1)
            else:
                print("no tracklets in frame {}".format(frame))

    labels = np.array(labels)
    imgs = np.array(imgs)

    split_idx = int(parameters['test_ratio'] * len(imgs))
    imgs_test = np.array(imgs[:split_idx])
    imgs_train = np.array(imgs[split_idx:])
    labels_test = np.array(labels[:split_idx])
    labels_train = np.array(labels[split_idx:])

    print("imgs TEST: {}, TRAIN: {}".format(imgs_test.shape, imgs_train.shape))
    print("labels TEST: {}, TRAIN: {}".format(labels_test.shape, labels_train.shape))

    with h5py.File(join(out_dir, 'descriptor_cnn_imgs_train.h5'), 'w') as hf:
        hf.create_dataset("data", data=imgs_train)
    with h5py.File(join(out_dir, 'descriptor_cnn_imgs_test.h5'), 'w') as hf:
        hf.create_dataset("data", data=imgs_test)
    with h5py.File(join(out_dir, 'descriptor_cnn_labels_train.h5'), 'w') as hf:
        hf.create_dataset("data", data=labels_train)
    with h5py.File(join(out_dir, 'descriptor_cnn_labels_test.h5'), 'w') as hf:
        hf.create_dataset("data", data=labels_test)
    open(join(out_dir, 'descriptor_cnn_params.yaml'), 'w').write(yaml.dump(parameters))


if __name__ == '__main__':
    # P_WD = '/Users/flipajs/Documents/wd/FERDA/april-paper/Cam1_clip'
    # P_WD = '/Users/flipajs/Documents/wd/FERDA/april-paper/Sowbug3-crop'
    # path = '/Users/flipajs/Documents/wd/FERDA/april-paper/5Zebrafish_nocover_22min'
    # path = '/Users/flipajs/Documents/wd/FERDA/april-paper/Camera3-5min'
    path = '../projects/Sowbug_deleteme2'
    # P_WD = '/Users/flipajs/Documents/wd/FERDA/zebrafish_new'
    project = Project(path)
    generate_reidentification_training_data(project, path)


