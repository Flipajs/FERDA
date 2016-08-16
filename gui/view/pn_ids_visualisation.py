import numpy as np
import cv2


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
                  'cross_thickness': 2,
                  'show_probabilities': True,
                  'prob_color': [255, 255, 255],
                  'prob_thickness': 3,
                  }



def draw(ids, P, N, probs=None, params=None):
    if params is None:
        params = default_params

    max_w = params['P_width'] * len(ids)
    max_h = params['P_height']

    img = np.zeros((max_h, max_w, 3), dtype=np.uint8) * 255

    w = 0
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
    img = img[:, :w, :].copy()

    return img

def draw_P(img, w, id_, params):
    new_w = w + params['P_width']
    img[0:params['P_height'], w:new_w, :] = params['colors'][id_]

    w += params['N_width']

    return new_w


def draw_N(img, w, id_, params):
    new_w = w + params['N_width']
    y1 = (params['P_height'] - params['N_height']) / 2
    y2 = params['N_height'] + y1

    img[y1:y2, w:new_w, :] = params['colors'][id_]

    # draw diagonal cross
    cv2.line(img, (w, y1), (new_w, y2), params['cross_color'], thickness=params['cross_thickness'])
    cv2.line(img, (new_w, y1), (w, y2), params['cross_color'], thickness=params['cross_thickness'])

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

    cv2.line(img, (mid_w, params['P_height']), (mid_w, params['P_height']-h), params['prob_color'], params['prob_thickness'])

    pass

if __name__ == '__main__':
    img = draw(set(range(6)), set([1, 2]), set([3, 5]), probs=[0.2, 0.1, 0.6, 0.05, 0.01, 0.04])
    cv2.imshow('img', img)
    cv2.waitKey(0)

    pass