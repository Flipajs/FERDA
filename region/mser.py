__author__ = 'fnaiser'

import cyMser
import cv2
import numpy
from region import Region


class Mser():
    def __init__(self, max_area=0.005, min_margin=5, min_area=5):
        self.mser = cyMser.PyMser()
        self.mser.set_min_margin(min_margin)
        self.mser.set_max_area(max_area)
        self.mser.set_min_size(min_area)

    def process_image(self, img, intensity_threshold=256):
        if len(img.shape) > 2:
            if img.shape[2] > 1:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img[:, :, 0]
        else:
            gray = img

        if intensity_threshold > 256:
            intensity_threshold = 256

        self.mser.process_image(gray, intensity_threshold)
        regions = self.mser.get_regions()

        regions = [Region(dr) for dr in regions]

        return regions