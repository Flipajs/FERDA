import numpy as np
from scipy import ndimage


def get_colormarks(img, cm_model):
    # img_t = transform_img_(img, cm_model)
    # img_t = img

    pos = np.asarray(img / cm_model.num_bins_v, dtype=np.int)
    labels = cm_model.get_labelling(pos)

    # TOOD: parameters
    ccs = get_ccs_(labels, bg=0, max_a=2000)
    return ccs


def match_cms_region(cms, r):
    pass


def get_ccs_(im, bg=-1, min_a=1, max_a=5000):
    import skimage

    labeled, num = skimage.measure.label(im, background=bg, return_num=True)

    # sizes = ndimage.sum(np.ones((im.shape[0], im.shape[1])), labeled, range(1, num + 1))

    ccs = []
    for i in range(num):
        ids = np.argwhere(labeled == i)
        if min_a < len(ids) < max_a:
            cc_label = im[ids[0, 0], ids[0, 1]]
            ccs.append((ids, cc_label))

    return ccs


def get_channels_255_(im, channels=[0, 1, 2]):
    irgb = irgb_transformation_(im)
    irg = irgb[:, :, channels]

    # irg[:, :, 0] = irg[:, :, 0] ** 0.5

    irg_255 = np.zeros(irg.shape)
    irg_255[:, :, 0] = irg[:, :, 0] / np.max(irg[:, :, 0])
    irg_255[:, :, 1] = irg[:, :, 1] / np.max(irg[:, :, 1])
    irg_255[:, :, 2] = irg[:, :, 2] / np.max(irg[:, :, 2])
    irg_255 = np.asarray(irg_255 * 255, dtype=np.uint8)

    return irg_255


def irgb_transformation_(im):
    I_NORM = 766 * 3 * 2
    irgb = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.double)

    irgb[:, :, 0] = np.sum(im, axis=2) + 1
    irgb[:, :, 1] = im[:, :, 0] / irgb[:, :, 0]
    irgb[:, :, 2] = im[:, :, 1] / irgb[:, :, 0]
    irgb[:, :, 3] = im[:, :, 2] / irgb[:, :, 0]

    irgb[:, :, 0] = irgb[:, :, 0] / I_NORM

    return irgb


def transform_img_(img, cm_model):
    if cm_model.im_space == 'irb':
        img_t = get_channels_255_(img, [0, 1, 3])
    elif cm_model.im_space == 'irg':
        img_t = get_channels_255_(img, [0, 1, 2])

    return img_t
