import cv2
import numpy as np

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

