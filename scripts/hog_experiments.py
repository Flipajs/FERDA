__author__ = 'flipajs'

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import data, color, exposure
from utils.video_manager import get_auto_video_manager
from core.project.project import Project
from core.region.mser import ferda_filtered_msers
import scipy

from PIL import Image
from scipy import ndimage
from pylab import *
from utils.drawing.points import draw_points

def get_mser(im, p):
    p.mser_parameters.max_area = 0.99
    msers = ferda_filtered_msers(np.asarray(im*255, dtype=np.uint8), p, 0)

    m = msers[0]
    ab = m.area() / np.pi
    cd = m.a_ * m.b_
    r = cd/ab

    b = ((r*m.a_*(m.b_**2))/m.a_)**0.5
    a = ab/b

    print "PREV: ", m.a_, m.b_
    print "NOW: ", a, b, np.rad2deg(m.theta_), m.theta_

    return msers[0]

def Haffine_from_points(fp,tp):
    """ find H, affine transformation, such that
        tp is affine transf of fp"""

    if fp.shape != tp.shape:
        raise RuntimeError, 'number of points do not match'

    #condition points
    #-from points-
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1))
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = dot(C1,fp)

    #-to points-
    m = mean(tp[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = dot(C2,tp)

    #conditioned points have mean zero, so translation is zero
    A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U,S,V = linalg.svd(A.T)

    #create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1)
    H = vstack((tmp2,[0,0,1]))

    #decondition
    H = dot(linalg.inv(C2),dot(H,C1))

    return H / H[2][2]

if __name__ == "__main__":
    p = Project()
    p.video_paths = ['/Users/flipajs/Documents/wd/c4_0h30m-0h33m.avi']
    vid = get_auto_video_manager(p)

    im = vid.next_frame()
    crop = im[746:801, 540:611, :]
    image = color.rgb2gray(crop)

    m = get_mser(image, p)
    crop = draw_points(crop, m.pts())

    head = np.array([m.a_*math.sin(-m.theta_), m.a_*math.cos(-m.theta_)]) + m.centroid()
    back = np.array([m.a_*math.sin(-m.theta_+np.pi), m.a_*math.cos(-m.theta_+np.pi)]) + m.centroid()
    print head, back
    crop[head[0], head[1], :] = (0, 0, 255)
    crop[back[0], back[1], :] = (0, 255, 0)

    # H = np.array([[1.4,0.05,-100],[0.05,1.5,-100],[0,0,1]])
    n, m = image.shape

    # H = Haffine_from_points(fp, tp)
    import cv2

    if True:
        srcP = np.float32(np.array([[0, 0], [m, 0], [m, n], [0, n]]))
        dstP = np.float32(np.array([[-30, 50], [m-50, -50], [m+80, n-100], [150, n+50]]))

        H = cv2.findHomography(srcP, dstP, 0)
        # M = np.float32([[0.8,0.5,10],[0,1,5]])
        im2 = cv2.warpAffine(image, H[0][0:2, :], (m, n))
        if True:
            cv2.imshow('dst', im2)
        # im2 = np.asarray(im2, dtype=np.uint8)
    # im2 = scipy.ndimage.affine_transform(image, H)

    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.axis('off')
    ax1.imshow(crop, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
