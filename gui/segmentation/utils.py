from __future__ import division
from __future__ import unicode_literals
from past.utils import old_div
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_shift(image, w=-1, h=-1, blur_kernel=3, blur_sigma=0.3, shift_x=2, shift_y=2):
    if w < 0 or h < 0:
        w, h, c = image.shape
  
    # prepare first image (original), make it grayscale and blurre
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img1, (blur_kernel, blur_kernel), blur_sigma)

    # create shift matrix
    M = np.float32([[1, 0, shift_x],[0, 1, shift_y]])
    # apply shift matrix
    img2 = cv2.warpAffine(image, M, (w, h))

    # prepare second image (tranlated), make it grayscale and blurred
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.GaussianBlur(img2, (blur_kernel, blur_kernel), blur_sigma)

    # get dif image
    dif = np.abs(np.asarray(img1, dtype=np.int32) - np.asarray(img2, dtype=np.int32))
    return dif

def get_avg(image):
    shift_up = get_shift(image, shift_x=-1, shift_y=0)
    shift_down = get_shift(image, shift_x=1, shift_y=0)
    shift_left = get_shift(image, shift_x=0, shift_y=-1)
    shift_right = get_shift(image, shift_x=0, shift_y=1)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    img_sum = shift_up + shift_down + shift_left + shift_right + img
    avg = old_div(img_sum, 5)
    return avg


def get_dif(image):
    shift_up = get_shift(image, shift_x=-1, shift_y=0)
    shift_down = get_shift(image, shift_x=1, shift_y=0)
    shift_left = get_shift(image, shift_x=0, shift_y=-1)
    shift_right = get_shift(image, shift_x=0, shift_y=1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dif_up = np.asarray(image, dtype=np.int32) - np.asarray(shift_up, dtype=np.int32)
    dif_down = np.asarray(image, dtype=np.int32) - np.asarray(shift_down, dtype=np.int32)
    dif_left = np.asarray(image, dtype=np.int32) - np.asarray(shift_left, dtype=np.int32)
    dif_right = np.asarray(image, dtype=np.int32) - np.asarray(shift_right, dtype=np.int32)

    difs = np.dstack((dif_up, dif_down, dif_left, dif_right))
    maxs = np.amax(difs, axis=2)
    mins = np.amin(difs, axis=2)

    diff = np.asarray(maxs, dtype=np.int32) - np.asarray(mins, dtype=np.int32)
    return maxs, mins, diff
