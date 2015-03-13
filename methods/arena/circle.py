__author__ = 'fnaiser'

import cv2
import numpy as np
from methods.arena.model import Model


class Circle(Model):
    def __init__(self, im_height, im_width):
        super(Circle, self).__init__(im_height, im_width)

    def set_circle(self, center, radius):
        self.mask_ = np.zeros((self.im_height, self.im_width), dtype=np.uint8)
        cv2.circle(self.mask_, (center[0], center[1]), radius, 255, -1)

        self.mask_idx_ = (self.mask_ == 0)




if __name__ == '__main__':
    im = cv2.imread('/Users/fnaiser/Documents/colormarktests/imgs/0.png')

    c = Circle(im.shape[0], im.shape[1])
    c.set_circle(np.array([140, 200]), 200)

    m = c.mask_image(im)
    cv2.imshow('test', m)
    cv2.waitKey(0)