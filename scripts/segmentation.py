from utils import video_manager

__author__ = 'filip@naiser.cz'

import cv2
import numpy as np
from scipy import ndimage

path = '/media/flipajs/Seagate Expansion Drive/IST - videos/compressed/bigLenses_colormarks1_test/'
orig_file = '/media/flipajs/Seagate Expansion Drive/IST - videos/bigLenses_colormarks1.avi'

path = '/Volumes/Seagate Expansion Drive/IST - videos/tests/segmentation/'
orig_file = '/Volumes/Seagate Expansion Drive/IST - videos/tests/output.avi'


dilation_iter_num = 15
thresh = 25

vid = video_manager.VideoManager(orig_file)

bg = cv2.imread(path+'bg.jpg')
bg_gray = gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

frame = 0
while(True):
    print frame

    img = vid.move2_next()
    if img is None:
        break


    img_orig = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = np.asarray(img_gray, dtype=np.int32)

    bg = np.asarray(bg_gray, dtype=np.int32)
    img = np.subtract(bg, img)
    idx = img[:,:] < 0
    img[idx] = 0

    img = np.asarray(img, dtype=np.uint8)
    img = ndimage.gaussian_filter(img, sigma=1)
    # arena = my_utils.RotatedRect(my_utils.Point(530, 530), my_utils.Size(910, 910), 0)
    # masked = my_utils.mask_out_arena(img, arena)

    img_bw = np.zeros(img.shape)

    idx = img[:,:] > thresh
    img_bw[idx] = 255

    img_bw = ndimage.binary_dilation(img_bw, iterations=dilation_iter_num).astype(img_bw.dtype)

    idx = img_bw[:,:] < 1
    img_orig[idx] = 255

    num = str(frame)
    while len(num) < 7:
        num = '0' + num

    # cv2.imshow("img", img_orig)
    # cv2.waitKey(0)

    cv2.imwrite(path+'frames/'+str(num)+'.png', img_orig, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    frame += 1