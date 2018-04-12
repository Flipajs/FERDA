import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
import math
import numpy.linalg as la


def get_colormarks(img, cm_model, min_a=0, max_a=20000):
    pos = np.asarray(img / cm_model.num_bins_v, dtype=np.int)
    labels = cm_model.get_labelling(pos)

    # TOOD: parameters
    ccs = get_ccs_(labels, bg=0, min_a=min_a, max_a=max_a)
    return ccs


def match_cms_region(cms, r, offset, thresh=2.0):
    cms_ = []
    cont = r.contour()

    # centroid test - colormark can be only near endpoints
    p_ = np.array([r.ellipse_major_axis_length()*math.sin(-r.theta_), r.ellipse_major_axis_length()*math.cos(-r.theta_)])
    endpoint1 = np.ceil(r.centroid() + p_)
    endpoint2 = np.ceil(r.centroid() - p_)

    thresh1 = r.ellipse_major_axis_length()
    thresh2 = (1/6.0) * np.pi

    for cm in cms:
        centroid = offset + np.sum(cm[0], 0) / cm[0].shape[0]
        if la.norm(centroid - endpoint1) > thresh1:
            if la.norm(centroid - endpoint2) > thresh1:
                continue

        v1 = endpoint1 - endpoint2
        v2 = endpoint1 - centroid

        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        th1 = abs(np.arctan2(sinang, cosang))

        v1 = endpoint2 - endpoint1
        v2 = endpoint2 - centroid
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        th2 = abs(np.arctan2(sinang, cosang))

        if th1 > thresh2 and th2 > thresh2:
            continue

        cms_.append(cm)

    cms = cms_
    cms_ = []

    for cm in cms:
        d_ = cdist(cm[0] + offset, cont)
        mins_ = np.min(d_, 0)

        min_ = min(mins_)

        if min_ <= thresh:
            cms_.append(cm)

    return cms_

def filter_cms(cms):
    cms_ = []
    for cm in cms:
        if len(cm[0]) > 15:
            cms_.append(cm)

    return cms_


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
