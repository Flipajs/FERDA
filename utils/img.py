__author__ = 'fnaiser'

import numpy as np
from PIL import ImageQt
from PyQt4 import QtGui
from utils.misc import get_settings

def get_safe_selection(img, y, x, height, width, fill_color=(255,255,255)):
    y = int(y)
    x = int(x)
    height = int(height)
    width = int(width)

    border = max(max(-y, -x), 0)

    channels = 1
    if len(img.shape) > 2:
        channels = img.shape[2]

    if len(fill_color) != channels:
        fill_color = 255

    h_ = img.shape[0] - (height + y)
    w_ = img.shape[1] - (width + x)

    border = max(border, max(max(-h_, -w_), 0))

    if border > 0:
        img_ = np.zeros((img.shape[0] + 2 * border, img.shape[1] + 2 * border, channels), dtype=img.dtype)
        img_ += fill_color
        img_[border:-border, border:-border] = img
        crop = np.ones((height, width, channels), dtype=img.dtype)
        crop *= fill_color

        y += border
        x += border
        crop = np.copy(img_[y:y+height, x:x+width, :])
    else:
        crop = np.copy(img[y:y+height, x:x+height, :])

    return crop

def get_pixmap_from_np_bgr(np_image):
    img_q = ImageQt.QImage(np_image.data, np_image.shape[1], np_image.shape[0], np_image.shape[1] * 3, 13)
    pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

    return pix_map

def get_roi(pts):
    """
    Returns ROI tupple (y, x, height, width) - Region Of Interest for given points

    :param pts:
    :return:
    """
    x = np.min(pts[:, 1])
    width = np.max(pts[:,1]) - x + 1
    y = np.min(pts[:,0])
    height = np.max(pts[:,0]) - y + 1

    return (y, x, height, width)

def avg_circle_area_color(im, y, x, radius):
    """
    computes average color in circle area given by pos and radius
    :param im:
    :param pos:
    :param radius:
    :return:
    """

    c = np.zeros((1, 3), dtype=np.double)
    num_px = 0
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            d = ((w - radius) ** 2 + (h - radius) ** 2) ** 0.5
            if d <= radius:
                num_px += 1
                c += im[y - radius + h, x - radius + w, :]

    print num_px
    c /= num_px

    return [c[0, 0], c[0, 1], c[0, 2]]

def get_igbr_normalised(im):
    igbr = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.double)

    igbr[:,:,0] = np.sum(im,axis=2) + 1
    igbr[:, :, 1] = im[:,:,0] / igbr[:,:,0]
    igbr[:,:,2] = im[:,:,1] / igbr[:,:,0]
    igbr[:,:,3] = im[:,:,2] / igbr[:,:,0]

    i_norm = (1/get_settings('igbr_i_weight', float)) * get_settings('igbr_i_norm', float)
    igbr[:,:,0] = igbr[:,:,0] / i_norm

    return igbr

def get_contour(pts):
    return -1
    # y_min, x_min = np.min(pts, axis=0)
    #
    # min_c = 100000
    # max_c = 0
    # min_r = 100000
    # max_r = 0
    #
    # if data == None:
    #     min_r = region['rle'][0]['line']
    #     max_r = region['rle'][-1]['line']
    #     for r in region['rle']:
    #         if min_c > r['col1']:
    #             min_c = r['col1']
    #         if max_c < r['col2']:
    #             max_c = r['col2']
    # else:
    #     for pt in data:
    #         if min_c > pt[0]:
    #             min_c = pt[0]
    #         if max_c < pt[0]:
    #             max_c = pt[0]
    #         if min_r > pt[1]:
    #             min_r = pt[1]
    #         if max_r < pt[1]:
    #             max_r = pt[1]
    #
    # rows = max_r - min_r
    # cols = max_c - min_c
    #
    # img = np.zeros((rows+1, cols+1), dtype=np.uint8)
    #
    # if data == None:
    #     for r in region['rle']:
    #         row = r['line'] - min_r
    #         col1 = r['col1'] - min_c
    #         col2 = r['col2'] - min_c
    #         img[row][col1:col2+1] = 255
    # else:
    #     for pt in data:
    #         row = pt[1] - min_r
    #         col1 = pt[0] - min_c
    #         #col2 = r['col2'] - min_c
    #         img[row][col1] = 255
    #
    #
    # #cv2.imshow("img", img)
    #
    # ret,thresh = cv2.threshold(img, 127, 255, 0)
    # try:
    #     _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # except:
    #     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #
    # cnt = []
    #
    # for c in contours:
    #     for pt in c:
    #         cnt.append(pt)
    #
    # img_cont = np.zeros((rows+1, cols+1), dtype=np.uint8)
    # pts = []
    # for p in cnt:
    #     img_cont[p[0][1]][p[0][0]] = 255
    #     pts.append([p[0][0] + min_c, p[0][1] + min_r])
    #
    # #cv2.imshow("test", img2)
    # #cv2.waitKey(0)
    #
    # return pts, img, img_cont, min_r, min_c