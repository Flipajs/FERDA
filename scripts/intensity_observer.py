from __future__ import division
from __future__ import unicode_literals
from builtins import range
from past.utils import old_div
__author__ = 'flipajs'

import cv2
from utils.video_manager import VideoManager, get_auto_video_manager
from core.project.project import Project
import numpy as np
import matplotlib.pyplot as plt


def stretch_intensity(im):
    plt.figure(1)

    step = 100
    im = im[400:-200, 500:-1, :]

    cv2.imwrite('/Users/flipajs/Documents/wd/sobel_orig.png', im)
    plt.subplot(311)
    plt.imshow(im)
    im = im[:,:,2]

    im[im > 100] = 100
    im *= 2

    plt.subplot(312)
    plt.imshow(im, cmap='jet')
    # plt.show()

    from skimage.filters import sobel, roberts
    # im2 = sobel(im)
    im2 = roberts(im)
    im2 = np.asarray((old_div(im2,np.max(im2))) * 255, dtype=np.uint8)
    cv2.imwrite('/Users/flipajs/Documents/wd/roberts.png', im2)

    plt.subplot(313)
    plt.imshow(im2, cmap='gray')

    plt.show()
    # cv2.imshow('im', im)
    # cv2.waitKey(0)


p = Project()
p.video_paths = ['/Users/flipajs/Documents/wd/c2.avi']
vid = get_auto_video_manager(p)

for i in range(1000):
    im = vid.next_frame()
    if not (i % 100):
        stretch_intensity(im)
