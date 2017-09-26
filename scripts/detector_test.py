
import sys
import cPickle as pickle


from core.settings import Settings as S_
from utils.misc import is_flipajs_pc, is_matejs_pc
from core.project.project import Project
import numpy as np
import matplotlib.pylab as plt

p = Project()

# This is development speed up process (kind of fast start). Runs only on developers machines...
# if is_flipajs_pc() and False:
wd = None
if is_flipajs_pc():
    # wd = '/Users/iflipajs/Documents/wd/FERDA/Cam1_rf'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/test6'
    # wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_rfs2'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1'
    wd = '/Users/flipajs/Documents/wd/FERDA/rep1-cam2'
    # wd = '/Users/flipajs/Documents/wd/FERDA/rep1-cam3'

    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'

    # wd = '/Users/flipajs/Documents/wd/FERDA/test'

if is_matejs_pc():
    # wd = '/home/matej/prace/ferda/10-15/'
    wd = '/home/matej/prace/ferda/10-15 (copy)/'

if wd is not None:
    p.load(wd)

from core.graph.region_chunk import RegionChunk
from utils.video_manager import get_auto_video_manager
from utils.img import get_img_around_pts
from utils.drawing.points import get_roi

import cv2

video = get_auto_video_manager(p)

ant_cascade = cv2.CascadeClassifier('/home/matej/prace/ferda/test/cascade.xml')

frame = 0
key = 0

fig = plt.figure()
closed = False


def handle_close(evt):
    global closed
    closed = True


def waitforbuttonpress():
    while plt.waitforbuttonpress(0.2) is None:
        if closed:
            return False
    return True

fig.canvas.mpl_connect('close_event', handle_close)

while True:
    img = video.get_frame(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_center = np.round(np.array(gray.shape)[::-1] / 2).astype(int)  # x, y
    # minSize=(56, 82), maxSize=(56, 82))
    detections = {}
    for angle in xrange(0, 180, 20):
        rot = cv2.getRotationMatrix2D(tuple(img_center), angle, 1.)
        img_rot = cv2.warpAffine(gray, rot, gray.shape[::-1])
        ants = ant_cascade.detectMultiScale(img_rot, 1.05, 5, minSize=(56, 82), maxSize=(56, 82))
        detections[angle] = []
        for (x, y, w, h) in ants:
            # [tl, tr, br, bl]
            corners = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]])
            corners_rotated = cv2.invertAffineTransform(rot).dot(np.hstack((corners, np.ones((4, 1)))).T).T
            detections[angle].append(corners_rotated)

    for angle, rotated_boxes in detections.iteritems():
        cv2.polylines(img, [np.array((corners)).astype(np.int32) for corners in rotated_boxes], True, (255, 0, 0))
    plt.imshow(img[::-1])
    if not waitforbuttonpress():
        break
    # cv2.imshow('img', img)
    # key = cv2.waitKey(0)
    frame += 100


# cv2.destroyAllWindows()

