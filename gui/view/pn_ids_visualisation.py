import numpy as np
import cv2
from PyQt4 import QtGui
from gui.img_controls.gui_utils import cvimg2qtpixmap

default_params = {'P_width': 20,
                  'P_height': 20,
                  'N_width': 13,
                  'N_height': 13,
                  'U_width': 10,
                  'U_height': 10,
                  # TODO: add more colors
                  'colors': [[255, 0, 0],
                             [0, 255, 0],
                             [0, 0, 255],
                             [255, 255, 0],
                             [255, 0, 255],
                             [0, 255, 255],
                             ],
                  'cross_color': [0, 0, 0],
                  'cross_thickness': 1,
                  'show_probabilities': True,
                  'prob_color': [255, 255, 255],
                  'prob_thickness': 1,
                  'prob_color_border': [0, 0, 0],
                  'prob_thickness_border': 2,
                  'N_darken_value': 60,
                  }


class PNIdsItem(QtGui.QGraphicsPixmapItem):
    def __init__(self, pixmap, id_=None, callback=None):
        super(PNIdsItem, self).__init__(pixmap)

        self.id_ = id_
        self.callback = callback
        self.setOpacity(1.0)

    def mousePressEvent(self, event):
        super(PNIdsItem, self).mousePressEvent(event)
        self.setOpacity(.1)

    def mouseReleaseEvent(self, event):
        super(PNIdsItem, self).mouseReleaseEvent(event)
        self.setOpacity(1.0)
        if self.callback is not None:
            self.callback(self.id_)


def get_pixmap_item(ids, P, N, tracklet_id=None, callback=None, probs=None, params=None, tracklet_len=0,
                    tracklet_ptr=0, tracklet_class_color=None):
    img = draw(ids, P, N, probs=probs, params=params, tracklet_len=tracklet_len, tracklet_ptr=tracklet_ptr,
               tracklet_class_color=tracklet_class_color)
    pix_map = cvimg2qtpixmap(img)

    p = PNIdsItem(pix_map, id_=tracklet_id, callback=callback)

    return p


def draw(ids, P, N, probs=None, params=None, tracklet_len=0, tracklet_ptr=0, tracklet_class_color=None):
    if params is None:
        params = default_params

    h = 4

    hh = 0
    if tracklet_len > 0:
        hh = 3

    max_w = params['P_width'] * len(ids) + 2
    max_h = h + params['P_height'] + hh

    img = np.zeros((max_h, max_w, 3), dtype=np.uint8) * 255
    if tracklet_class_color:
        img[:h, :, :] = (tracklet_class_color[2], tracklet_class_color[1], tracklet_class_color[0])

    if tracklet_len > 0:
        img[-hh:, :min(max_w, tracklet_len), :] = (255, 255, 0)
        img[-hh:, max(0, tracklet_ptr-1):min(max_w, tracklet_ptr+1), :] = (0, 0, 255 )

    w = 1
    for id_ in ids:
        old_w = w

        if id_ in P:
            # present
            w = draw_P(img, w, id_, params)
        elif id_ in N:
            # not present
            w = draw_N(img, w, id_, params)
        else:
            # unknown
            w = draw_U(img, w, id_, params)

        if params['show_probabilities'] and probs is not None:
            show_probs(img, old_w, w, probs[id_], params)

    # crop it...
    img = img[:, :w+1, :].copy()

    return img

def draw_P(img, w, id_, params):
    new_w = w + params['P_width']
    img[1:params['P_height']-1, w:new_w, :] = params['colors'][id_]

    w += params['N_width']

    return new_w


def draw_N(img, w, id_, params):
    from utils.visualization_utils import get_contrast_color

    new_w = w + params['N_width']
    y1 = (params['P_height'] - params['N_height']) / 2
    y2 = params['N_height'] + y1

    c = params['colors'][id_]
    v = params['N_darken_value']
    if v > 0:
        c[0] = max(0, c[0] - v)
        c[1] = max(0, c[1] - v)
        c[2] = max(0, c[2] - v)

    img[y1:y2, w:new_w, :] = c

    # draw diagonal cross
    cv2.line(img, (w, y1), (new_w, y2), get_contrast_color(c[0], c[1], c[2]), thickness=params['cross_thickness'])
    cv2.line(img, (new_w, y1), (w, y2), get_contrast_color(c[0], c[1], c[2]), thickness=params['cross_thickness'])

    return new_w


def draw_U(img, w, id_, params):
    new_w = w + params['U_width']
    y1 = (params['P_height'] - params['U_height']) / 2
    y2 = params['U_height'] + y1
    img[y1:y2, w:new_w, :] = params['colors'][id_]

    return new_w


def show_probs(img, old_w, w, prob, params):
    mid_w = w + (old_w - w) / 2

    h = int(round(params['P_height'] * prob))

    cv2.line(img, (mid_w, params['P_height']), (mid_w, params['P_height']-h), params['prob_color_border'], params['prob_thickness_border'])
    cv2.line(img, (mid_w, params['P_height']), (mid_w, params['P_height']-h), params['prob_color'], params['prob_thickness'])

    pass

if __name__ == '__main__':
    img = draw(set(range(6)), set([1, 2]), set([3, 5]), probs=[0.2, 0.1, 0.6, 0.05, 0.01, 0.04])
    cv2.imshow('img', img)
    cv2.waitKey(0)

    pass