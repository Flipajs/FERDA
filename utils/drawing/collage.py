import numpy as np
import cv2
from math import ceil, floor


def create_collage_rows(data, cols, item_h, item_w, upscale=False, fill=False):
    """
    fill means - fit to the size (item_h, item_w)

    Args:
        data:
        cols:
        item_h:
        item_w:
        upscale:
        fill:

    Returns:

    """
    row = 0
    col = 0

    rows = int(ceil(len(data) / float(cols)))
    result = np.zeros((rows * item_h, cols*item_w, 3), dtype=np.uint8)

    for im in data:
        h, w, _ = im.shape

        new_h, new_w = item_h, item_w
        if not fill:
            f = min(item_h / float(h), item_w / float(w))

            new_h = int(floor(h * f))
            new_w = int( floor(w * f))

        if upscale:
            im = cv2.resize(im, (new_w, new_h))
        else:
            if h > item_h or w > item_w:
                im = cv2.resize(im, (new_w, new_h))

        h_missing = item_h - im.shape[0]
        w_missing = item_w - im.shape[1]
        result[item_h*row:item_h*(row+1) - h_missing,
               item_w*col:item_w*(col+1) - w_missing, :] = im.copy()

        col += 1
        if col == cols:
            col = 0
            row += 1

    return result
