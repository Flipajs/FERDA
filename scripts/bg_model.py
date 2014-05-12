__author__ = 'flipajs'

import video_manager
import numpy as np
import cv2

vid = video_manager.VideoManager("/home/flipajs/Dropbox/PycharmProjects/data/eight/eight.m4v")
#vid = video_manager.VideoManager("/home/flipajs/Dropbox/PycharmProjects/data/NoPlasterNoLid800/NoPlasterNoLid800.m4v")

bg = None
i = -1
while True:
    i += 1
    img = vid.next_img()

    if img is None:
        break

    if i % 50 != 0:
        continue

    if bg != None:
        bg = np.maximum(bg, img)
    else:
        bg = img



cv2.imshow("BG", bg)
cv2.waitKey(0)
cv2.imwrite('bg.png', bg)