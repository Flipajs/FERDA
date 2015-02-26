__author__ = 'fnaiser'

import numpy as np


def get_safe_selection(img, y, x, height, width, fill_color=(0,0,0)):
    y = int(y)
    x = int(x)
    height = int(height)
    width = int(width)

    border = max(max(-y, -x), 0)

    channels = 1
    if len(img.shape) > 2:
        channels = img.shape[2]

    h_ = img.shape[0] - (height + y)
    w_ = img.shape[1] - (width + x)

    border = max(border, max(max(-h_, -w_), 0))

    if border > 0:
        img_ = np.zeros((img.shape[0] + 2 * border, img.shape[1] + 2 * border, channels), dtype=img.dtype)
        img_ += fill_color
        img_[border:-border, border:-border] = img
        crop = np.ones((height, width, channels), dtype=img.dtype)
        if channels < 3:
            fill_color = fill_color[0]

        crop *= fill_color

        y += border
        x += border
        crop = np.copy(img_[y:y+height, x:x+width, :])
    else:
        crop = np.copy(img[y:y+height, x:x+height, :])

    return crop


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