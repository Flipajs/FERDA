__author__ = 'fnaiser'

import numpy as np

class BGModel():
    def __init__(self, bg):
        self.bg_model_ = bg

    def bg_subtraction(self, img):
        processed = np.subtract(np.asarray(self.bg_model_, dtype=np.int32), np.asarray(img, dtype=np.int32))
        processed[processed < 0] = 0
        processed[processed > 255] = 255
        processed = np.asarray(processed, dtype=np.uint8)
        processed = np.invert(processed)

        return processed

    def img(self):
        return np.copy(self.bg_model_)

    def update(self, img):
        self.bg_model_ = np.copy(img)