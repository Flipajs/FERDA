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
from utils.roi import get_roi


def get_mser(im, p):
    p.mser_parameters.max_area = 1000000
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
    roi = get_roi(r.pts())
    # tl = roi.top_left_corner()-np.array([1, 1])
    # br = roi.bottom_right_corner()+np.array([1, 1])

    im_ = np.ones_like(im)
    im_[r.pts()[:, 0], r.pts()[:, 1]] = im[r.pts()[:, 0], r.pts()[:, 1]]

    # crop = im[tl[0]:br[0], tl[1]:br[1]].copy()

    p_ = np.array([r.a_*math.sin(-r.theta_), r.a_*math.cos(-r.theta_)])
    head = np.ceil(r.centroid() + p_) + np.array([1, 1])
    back = np.ceil(r.centroid() - p_) - np.array([1, 1])

    if head[0] < back[0]:
        head, back = back, head
    
    b_ = r.b_*2.5
    p_ = np.array([b_*math.sin(-r.theta_+np.pi+np.pi/2), b_*math.cos(-r.theta_+np.pi+np.pi/2)])
    tl_c = back + p_
    tr_c = head + p_
    bl_c = back - p_
    br_c = head - p_

    src_pts = np.float32(np.array([tl_c, tr_c, br_c, bl_c]))
    # srcP = np.float32(np.array([[0, 0], [dst_h, 0], [dst_h, dst_w], [0, dst_w]]))
    dst_pts = np.float32(np.array([[0, 0], [0, dst_w], [dst_h, dst_w], [dst_h, 0]]))
    # dstP = np.float32(np.array([[-30, 50], [m-50, -50], [m+80, n-100], [150, n+50]]))

    src_pts[:, 0], src_pts[:, 1] = src_pts[:, 1].copy(), src_pts[:, 0].copy()
    dst_pts[:, 0], dst_pts[:, 1] = dst_pts[:, 1].copy(), dst_pts[:, 0].copy()

    # src_pts = np.float32(np.array([[tl_c[1], tl_c[0]], [tr_c[1], tr_c[0]], [br_c[1], br_c[0]], [bl_c[1], bl_c[0]]]))
    # # dst_pts = np.float32(np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]]))
    # dst_pts = np.float32(np.array([[0, dst_h], [dst_w, dst_h], [dst_w, 0], [0, 0]]))

    # src_pts[:, 0], src_pts[:, 1] = src_pts[:, 1].copy(), src_pts[:, 0].copy()
    # dst_pts[:, 0], dst_pts[:, 1] = dst_pts[:, 1].copy(), dst_pts[:, 0].copy()

    A = cv2.getAffineTransform(src_pts[0:3], dst_pts[0:3])
    # A = np.array([[1, 0, 30], [0, 1, 50]], dtype=np.float32)

    # crop_ = np.asarray(crop_*255, dtype=np.uint8)
    # im2 = cv2.warpAffine(crop, A, (dst_w, dst_h))
    # A[0, 2] = dst_h
    # A[1, 2] = dst_w

    im2 = cv2.warpAffine(im_, A, (dst_w, dst_h))

    return im2

def hogs_test(p, chunks):
    hogs = {}
    for ch in chunks:
        with open(p.working_directory+'/chunk'+str(ch)+'/hogs.pkl', 'rb') as f:
            up = pickle.Unpickler(f)
            hogs[ch] = up.load()

    for compare_with in chunks:
        right = 0
        wrong = 0

        for time_step in [20]:
            for t in range(0, 500-time_step):
                try:
                    h = hogs[compare_with][t]

                    best_match_d = np.inf
                    best_match_ch = -1

                    for ch in chunks:
                        d = np.linalg.norm(h-hogs[ch][t+time_step])
                        if d < best_match_d:
                            best_match_d = d
                            best_match_ch = ch

                    if best_match_ch == compare_with:
                        right += 1
                    else:
                        wrong += 1

                except KeyError:
                    pass

        if right+wrong:
            print compare_with, "#RIGHT: ", right, "#WRONG: ", wrong, "SR: ", round(right / float(right+wrong), 2)
        else:
            print "none"


def hogs_test2(p, chunks):
    hogs = {}
    for ch in chunks:
        with open(p.working_directory+'/chunk'+str(ch)+'/hogs.pkl', 'rb') as f:
            up = pickle.Unpickler(f)
            hogs[ch] = up.load()

    search_range = 7
    k_best = 5

    for compare_with in chunks:
        right = 0
        wrong = 0

        for time_step in [50]:
            for t in range(search_range, 500-time_step):
                try:
                    h = hogs[compare_with][t]

                    distances = []

                    for ch in chunks:
                        for t_ in range(t+time_step-search_range, t+time_step+search_range):
                            d = np.linalg.norm(h-hogs[ch][t_])
                            distances.append((ch, d))

                    distances = sorted(distances, key=lambda x: x[1])

                    v = 0
                    for i in range(k_best):
                        if distances[i][0] == compare_with:
                            v += 1

                    if v >= 3:
                        right += 1
                    else:
                        wrong += 1

                except KeyError:
                    pass

        if right+wrong:
            print compare_with, "#RIGHT: ", right, "#WRONG: ", wrong, "SR: ", round(right / float(right+wrong), 2)
        else:
            print "none"

def head_test(im_):
    """
    try if left half of image has more white pixels then right one if yes, flip image so the head is in right part of image.
    :param im_:
    :return:
    """
    h, w = im_.shape

    left_ = im_[:, 0:w/2]
    right_ = im_[:, w/2:]

    left_ = np.sum(left_[left_ == 255])
    right_ = np.sum(right_[right_ == 255])

    if left_ > right_:
        im_ = np.fliplr(im_)

    return im_

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    p = Project()
    p.working_directory = '/Users/flipajs/Documents/wd/c4'
    p.video_paths = ['/Users/flipajs/Documents/wd/c4_0h30m-0h33m.avi']
    # p.load('/Users/flipajs/Documents/wd/c4/c4.fproj')

    # for i in range(3, 10):
    #     for j in range(i+1, 10):
    #         for k in range(j+1, 10):
    #             print i, "vs", j, "vs ", k
    #             hogs_test2(p, [i, j, k])

    for i in range(3, 10):
        for j in range(i+1, 10):
            print i, "vs", j
            hogs_test2(p, [i, j])

    # hogs_test(p, [4, 5])
    # hogs_test(p, [4, 5])
    # hogs_test(p, [4, 5])

    if False:
        # solver = p.solver
        solver = None
        regions = get_regions(p, solver, 0, 500)

        vid = get_auto_video_manager(p)

        for chunk_id in range(3, 10):
            print chunk_id
            hogs = {}

            import os
            try:
                os.mkdir(p.working_directory+'/chunk'+str(chunk_id))
            except OSError:
                pass

            for f in range(500):
                im = vid.get_frame(f)
                gray = color.rgb2gray(im)

                reg = None
                for r in regions[f]:
                    if r['chunk_id'] == chunk_id:
                        reg = r['region']

                if not reg:
                    continue

                im_ = warp_region(reg, gray)
                im_ = head_test(im_)

                cv2.imwrite(p.working_directory+'/chunk'+str(chunk_id)+'/'+str(f)+'.png', np.asarray(im_*255, dtype=np.uint8))

                fd, hog_image = hog(im_, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(1, 1), visualise=True)

                hogs[f] = fd

                # im_ = np.asarray(im_*255, dtype=np.uint8)
                # cv2.imshow('im_', im_)
                # cv2.moveWindow('im_', 0, 0)
                # cv2.waitKey(0)

            with open(p.working_directory+'/chunk'+str(chunk_id)+'/hogs.pkl', 'wb') as f:
                p_ = pickle.Pickler(f, -1)
                p_.dump(hogs)

        # im = vid.next_frame()
        #
        # fd, hog_image = hog(im2, orientations=9, pixels_per_cell=(8, 8),
        #                     cells_per_block=(1, 1), visualise=True)
        #
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))
        #
        # ax1.axis('off')
        # ax1.imshow(crop, cmap=plt.cr.gray)
        # ax1.set_title('Input image')
        #
        # ax2.axis('off')
        # ax2.imshow(im2, cmap=plt.cr.gray)
        # ax2.set_title('warped')
        #
        # # Rescale histogram for better display
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
        #
        # ax3.axis('off')
        # ax3.imshow(hog_image_rescaled, cmap=plt.cr.gray)
        # ax3.set_title('Histogram of Oriented Gradients')
        # plt.show()

    app.exec_()
    app.deleteLater()
    sys.exit()