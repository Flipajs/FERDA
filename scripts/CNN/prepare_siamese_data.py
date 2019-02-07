from __future__ import print_function
from __future__ import print_function
import numpy as np
import tqdm
from cachetools import LRUCache
from tqdm import tqdm
import random
import h5py
from os.path import join
from core.project.project import Project
from utils.img import get_safe_selection
from utils.img import apply_ellipse_mask

# TODO: project parameter?
OFFSET = 45
ELLIPSE_DILATION = 10
MASK_SIGMA = 10
BATCH_SIZE = 500
APPLY_ELLIPSE = True
ALIGN = True
NUM_EXAMPLES = 5000
TEST_RATIO = .1


def get_region_image(region, vm, offset=45, add_ellipse_mask=True, mask_sigma=10, ellipse_dilation=10, align=True):
    global cache
    img = cache[region.frame()]
    # img = vm.get_frame(region.frame())
    crop = get_region_crop(region, img, add_ellipse_mask, ellipse_dilation, mask_sigma, offset, align)

    return crop


def get_region_crop(region, img, add_ellipse_mask, ellipse_dilation, mask_sigma, offset, align=True):
    y, x = region.centroid()
    crop = get_safe_selection(img, y - offset, x - offset, 2 * offset, 2 * offset)
    if add_ellipse_mask:
        crop = apply_ellipse_mask(region, crop, mask_sigma, ellipse_dilation)

    if align:
        flip = 180*random.randint(0, 1)
        import imutils
        angle = int(region.theta_ * 57.295 + flip)
        crop = imutils.rotate(crop, -angle)

    return crop


def generate_reidentification_training_data(project_path, out_dir):
    LINEAR = False
    np.set_printoptions(precision=2)

    p = Project(project_path)
    vm = p.get_video_manager()
    cache = LRUCache(maxsize=5000, missing=lambda x: vm.get_frame(x))

    if LINEAR:
        for i in tqdm(range(p.num_frames())):
            cache[i]

    imgs = []
    labels = []
    i = 0
    with tqdm(total=NUM_EXAMPLES) as pbar:
        while i < NUM_EXAMPLES:
            frame = random.randint(0, vm.total_frame_count())

            tracklets = filter(lambda x: x.is_single(), p.chm.tracklets_in_frame(frame))
            if len(tracklets) > 1:
                trackletA, trackletB = random.sample(tracklets)
                regionA1 = trackletA.get_random_region()
                regionA2 = trackletA.get_random_region()
                regionB = trackletB.get_random_region()

                cropA1 = get_region_image(regionA1, vm, offset=OFFSET, add_ellipse_mask=APPLY_ELLIPSE,
                                          mask_sigma=MASK_SIGMA, ellipse_dilation=ELLIPSE_DILATION, align=ALIGN)
                cropA2 = get_region_image(regionA2, vm, offset=OFFSET, add_ellipse_mask=APPLY_ELLIPSE,
                                          mask_sigma=MASK_SIGMA, ellipse_dilation=ELLIPSE_DILATION, align=ALIGN)
                cropB = get_region_image(regionB, vm, offset=OFFSET, add_ellipse_mask=APPLY_ELLIPSE,
                                         mask_sigma=MASK_SIGMA, ellipse_dilation=ELLIPSE_DILATION, align=ALIGN)

                imgs += [[cropA1, cropA2]]
                imgs += [[cropA1, cropB]]
                labels += [1, 0]

                i += 1
                pbar.update(1)
            else:
                print("no tracklets in frame {}".format(frame))

    labels = np.array(labels)
    imgs = np.array(imgs)

    split_idx = int(TEST_RATIO * len(imgs))
    imgs_test = np.array(imgs[:split_idx])
    imgs_train = np.array(imgs[split_idx:])
    labels_test = np.array(labels[:split_idx])
    labels_train = np.array(labels[split_idx:])

    print("imgs TEST: {}, TRAIN: {}".format(imgs_test.shape, imgs_train.shape))
    print("labels TEST: {}, TRAIN: {}".format(labels_test.shape, labels_train.shape))

    if ALIGN:
        aligned = '_aligned'
    else:
        aligned = ''
    with h5py.File(join(out_dir, 'descriptor_cnn_imgs_train{}.h5'.format(aligned)), 'w') as hf:
        hf.create_dataset("data", data=imgs_train)
    with h5py.File(join(out_dir, 'descriptor_cnn_imgs_test{}.h5'.format(aligned)), 'w') as hf:
        hf.create_dataset("data", data=imgs_test)
    with h5py.File(join(out_dir, 'descriptor_cnn_labels_train{}.h5'.format(aligned)), 'w') as hf:
        hf.create_dataset("data", data=labels_train)
    with h5py.File(join(out_dir, 'descriptor_cnn_labels_test{}.h5'.format(aligned)), 'w') as hf:
        hf.create_dataset("data", data=labels_test)


if __name__ == '__main__':
    # P_WD = '/Users/flipajs/Documents/wd/FERDA/april-paper/Cam1_clip'
    # P_WD = '/Users/flipajs/Documents/wd/FERDA/april-paper/Sowbug3-crop'
    # path = '/Users/flipajs/Documents/wd/FERDA/april-paper/5Zebrafish_nocover_22min'
    # path = '/Users/flipajs/Documents/wd/FERDA/april-paper/Camera3-5min'
    path = '../projects/Sowbug_deleteme2'
    # P_WD = '/Users/flipajs/Documents/wd/FERDA/zebrafish_new'
    generate_reidentification_training_data(path, path)


