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
import cv2
from PyQt4 import QtGui, QtCore
import sys
import cPickle as pickle
from utils.roi import ROI


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

def get_regions(project, solver, from_t, to_t):
    try:
        with open(project.working_directory+'/regions.pkl', 'rb') as f:
            up = pickle.Unpickler(f)
            reconstructed = up.load()

    except IOError:
        nodes = solver.nodes_in_t[0]
        nodes = sorted(nodes, key=lambda x: x.area())

        from gui.statistics.region_reconstruction import RegionReconstruction
        rr = RegionReconstruction(project, solver)
        frames = range(from_t, to_t)
        reconstructed = rr.reconstruct_regions(frames)

        with open(project.working_directory+'/regions.pkl', 'wb') as f:
            p = pickle.Pickler(f, -1)
            p.dump(reconstructed)

        print "RECONSTRUCTION DONE"

    return reconstructed

def warp_region(r, im, dst_h=16, dst_w=48):
    roi = ROI(r)
    tl = roi.top_left_corner()-np.array([1, 1])
    br = roi.bottom_right_corner()+np.array([1, 1])
    crop = im[tl[0]:br[0], tl[1]:br[1], :].copy()

    p_ = np.array([r.a_*math.sin(-r.theta_), r.a_*math.cos(-r.theta_)])
    head = np.ceil(r.centroid() + p_) + np.array([1, 1])
    back = np.ceil(r.centroid() - p_) - np.array([1, 1])
    
    b_ = r.b_*2.5
    p_ = np.array([b_*math.sin(-r.theta_+np.pi+np.pi/2), b_*math.cos(-r.theta_+np.pi+np.pi/2)])
    tl_c = back + p_
    tr_c = head + p_
    bl_c = back - p_
    br_c = head - p_

    src_pts = np.float32(np.array([[tl_c[1], tl_c[0]], [tr_c[1], tr_c[0]], [br_c[1], br_c[0]], [bl_c[1], bl_c[0]]]))
    dst_pts = np.float32(np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]]))

    A = cv2.getAffineTransform(src_pts[0:3], dst_pts[0:3])

    im_ = np.zeros_like(crop)
    im_[r.pts()[:, 0], r.pts()[:, 1]] = crop[r.pts()[:, 0], r.pts()[:, 1]]

    im2 = cv2.warpAffine(im_, A, (dst_w, dst_h))

    return im2

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    p = Project()
    p.working_directory = '/Users/flipajs/Documents/wd/c4'
    p.video_paths = ['/Users/flipajs/Documents/wd/c4_0h30m-0h33r.avi']
    # p.load('/Users/flipajs/Documents/wd/c4/c4.fproj')

    # solver = p.saved_progress['solver']
    solver = None
    get_regions(p, solver, 0, 500)

    vid = get_auto_video_manager(p)

    im = vid.next_frame()
    # crop = im[746:801, 540:611, :]
    # image = color.rgb2gray(crop)

    # m = get_mser(image, p)
    # crop = draw_points(crop, r.pts())

    # p_ = np.array([r.a_*math.sin(-r.theta_), r.a_*math.cos(-r.theta_)])
    # head = np.ceil(r.centroid() + p_)
    # back = np.ceil(r.centroid() - p_)
    # print head, back
    # crop[r.centroid_[0], r.centroid_[1]] = (255, 255, 0)

    # dst_h = 8*2
    # dst_w = 8*6

    # head += np.array([1, 1])
    # back -= np.array([1, 1])

    # crop[head[0], head[1], :] = (0, 0, 255)
    # crop[back[0], back[1], :] = (0, 255, 0)

    # b_ = r.b_*2.5
    # p_ = np.array([b_*math.sin(-r.theta_+np.pi+np.pi/2), b_*math.cos(-r.theta_+np.pi+np.pi/2)])
    # tl_c = back + p_
    # tr_c = head + p_
    # bl_c = back - p_
    # br_c = head - p_
    # crop[tl_c[0], tl_c[1], :] = (255, 0, 0)
    # crop[tr_c[0], tr_c[1], :] = (255, 0, 0)
    # crop[bl_c[0], bl_c[1], :] = (255, 0, 0)
    # crop[br_c[0], br_c[1], :] = (255, 0, 0)

    if True:
        # src_pts = np.float32(np.array([tl_c, tr_c, br_c, bl_c]))
        # # srcP = np.float32(np.array([[0, 0], [dst_h, 0], [dst_h, dst_w], [0, dst_w]]))
        # dst_pts = np.float32(np.array([[0, 0], [0, dst_w], [dst_h, dst_w], [dst_h, 0]]))
        # # dstP = np.float32(np.array([[-30, 50], [m-50, -50], [m+80, n-100], [150, n+50]]))

        # src_pts[:, 0], src_pts[:, 1] = src_pts[:, 1].copy(), src_pts[:, 0].copy()
        # dst_pts[:, 0], dst_pts[:, 1] = dst_pts[:, 1].copy(), dst_pts[:, 0].copy()

        # H = cv2.findHomography(dst_pts, dst_pts, 0)

        # A = cv2.getAffineTransform(src_pts[0:3], dst_pts[0:3])
        # P = cv2.getPerspectiveTransform(dst_pts, src_pts)

        # tl_c = np.array([tl_c[0], tl_c[1], 1])
        # print np.dot(A, tl_c.T)

        # im2 = cv2.warpAffine(crop, H[0][0:2, :], (200, 200))
        # im_ = np.zeros_like(image)
        # # TODO: speed up...
        # # for pt in r.pts():
        # #     im_[pt[0], pt[1]] = image[pt[0], pt[1]]
        # im_[r.pts()[:, 0], r.pts()[:, 1]] = image[r.pts()[:, 0], r.pts()[:, 1]]
        #
        # # image = np.ma.masked_array(image, mask=r.pts())
        #
        # im2 = cv2.warpAffine(im_, A, (dst_w, dst_h))
        # im2 = cv2.warpPerspective(crop, P, (200, 200))

        # new_tl = np.dot(tl_c, A)
        # print new_tl
        # im2 = im2[tl_c[0]:tl_c[0]+dst_h+1, tl_c[1]:tl_c[1]+dst_w+1]

    fd, hog_image = hog(im2, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualise=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))

    ax1.axis('off')
    ax1.imshow(crop, cmap=plt.cr.gray)
    ax1.set_title('Input image')

    ax2.axis('off')
    ax2.imshow(im2, cmap=plt.cr.gray)
    ax2.set_title('warped')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax3.axis('off')
    ax3.imshow(hog_image_rescaled, cmap=plt.cr.gray)
    ax3.set_title('Histogram of Oriented Gradients')
    plt.show()

    app.exec_()
    app.deleteLater()
    sys.exit()