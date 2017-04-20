import numpy as np
from libs import color_naming
from utils import feature_descriptor


def colornames_descriptor(im, block_division=(2, 2), pyramid_levels=3, histogram_density=False):
    """

    Parameters
    ----------
    im : np.array, image with 3 channels in RGB order (be aware that cv2.imread() returns BGR ...)

    for further details see utils.py - feature_descriptor parameters
    block_division : tuple(uint, uint)
    pyramid_levels : uint
    histogram_density : bool

    Returns
    -------

    """

    # get color naming
    cm = color_naming.im2colors(im)

    # it is given by 11 color names
    histogram_num_bins = 11

    return feature_descriptor(cm, block_division=block_division, pyramid_levels=pyramid_levels,
                              histogram_num_bins=histogram_num_bins, histogram_density=histogram_density)


if __name__ == '__main__':
    from scipy.misc import imread

    im = imread('data/car.jpg')

    f = colornames_descriptor(im, histogram_density=True)
    print f