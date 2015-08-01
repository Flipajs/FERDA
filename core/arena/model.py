__author__ = 'fnaiser'

import numpy as np


class Model(object):
    def __init__(self, im_height, im_width):
        self.im_width = im_width
        self.im_height = im_height
        self.mask_ = None
        # indices of px to fill with color (out of arena)
        self.mask_idx_ = None
        # might be used for softening borders...
        self.weights_ = None

    def mask_image(self, img, fill=(255, 255, 255)):
        processed = np.copy(img)
        processed[self.mask_idx_] = fill

        #TODO if self.weights:

        return processed