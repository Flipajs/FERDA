__author__ = 'filip@naiser.cz'

import cv2

try:
    cv_vidWriter = cv2.VideoWriter_fourcc
except:
    cv_vidWriter = cv2.cv.CV_FOURCC

try:
    cv_CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
except:
    cv_CAP_PROP_POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES

try:
    cv_CAP_PROP_POS_MSEC = cv2.CAP_PROP_POS_MSEC
except:
    cv_CAP_PROP_POS_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC

try:
    cv_CAP_PROP_FPS = cv2.CAP_PROP_FPS
except:
    cv_CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS

try:
    cv_CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
except:
    cv_CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT