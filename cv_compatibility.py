__author__ = 'flipajs'

import cv2

try:
    cv_vidWriter = cv2.VideoWriter_fourcc
except:
    cv_vidWriter = cv2.cv.CV_FOURCC