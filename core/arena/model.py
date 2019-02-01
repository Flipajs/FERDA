__author__ = 'fnaiser'

import numpy as np
import cv2


class Model(object):
    def __init__(self, im_height, im_width):
        self.im_width = im_width
        self.im_height = im_height
        self.mask_ = None
        # indices of px to fill with color (out of arena)
        self.mask_idx_ = None
        # might be used for softening borders...
        self.weights_ = None
        self.mask_filename = None

    def save_mask(self, mask_filename):
        if self.mask_ is not None:
            cv2.imwrite(mask_filename, self.mask_)

    def load_mask(self, mask_filename):
        self.mask_ = cv2.imread(mask_filename)
        self.mask_idx_ = (self.mask_ == 0)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['mask_']
        del state['mask_idx_']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def mask_image(self, img, fill=(255, 255, 255)):
        if len(img.shape) == 2:
            fill = fill[0]
        elif img.shape[2] == 1:
            fill = fill[0]

        processed = np.copy(img)
        processed[self.mask_idx_] = fill
        #
        # #TODO if self.weights:
        #
        return processed
