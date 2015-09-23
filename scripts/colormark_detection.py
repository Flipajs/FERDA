__author__ = 'flipajs'

from utils.video_manager import VideoManager
import cv2
import numpy as np
import matplotlib.pyplot as plt

I_NORM = 766 * 3 * 2


def igbr_transformation(im):
    igbr = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.double)

    igbr[:,:,0] = np.sum(im,axis=2) + 1
    igbr[:, :, 1] = im[:,:,0] / igbr[:,:,0]
    igbr[:,:,2] = im[:,:,1] / igbr[:,:,0]
    igbr[:,:,3] = im[:,:,2] / igbr[:,:,0]

    igbr[:,:,0] = igbr[:,:,0] / I_NORM

    return igbr

if __name__ == "__main__":
    vid = VideoManager('/Users/flipajs/Documents/wd/C210min.avi')
    im = vid.next_frame()

    igbr = igbr_transformation(im)

    vis = im.copy()

    mser = cv2.MSER(_min_area=20, _min_margin=0.001, _edge_blur_size=0,
                    _delta=10, _max_variation=1.0, _min_diversity=0.1, _max_area=1000,
                    _max_evolution=20000)
    im__ = igbr[:,:,1:4].copy()
    im__ = np.asarray(255-(im__*255), dtype=np.uint8)
    vis = im__.copy()

    # regions = mser.detect(im[:,:,0])
    regions = mser.detect(im__[:,:,0])
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    cv2.imshow('img', vis)
    cv2.waitKey(0)

    cols = 3
    rows = 3
    fig = plt.figure()

    i = 1
    plt.subplot(int(str(cols)+str(rows)+str(i)))
    im_ = im.copy()
    im_[:,:,0] = im[:,:,2]
    im_[:,:,2] = im[:,:,0]
    plt.imshow(im_)

    for i in range(2, 6):
        plt.subplot(int(str(cols)+str(rows)+str(i)))
        plt.imshow(igbr[:, :, i-2], cmap='gray')

    from skimage import color
    lab = color.rgb2lab(im)

    for i in range(6, 9):
        plt.subplot(int(str(cols)+str(rows)+str(i)))
        plt.imshow(lab[:, :, i-6], cmap='gray')

    plt.show()
    plt.waitforbuttonpress()