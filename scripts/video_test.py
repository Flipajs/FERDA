__author__ = 'flipajs'

import cv2
import numpy as np
from viewer import video_manager

# vid_file = '/home/flipajs/Dropbox/PycharmProjects/ants/scripts/my_output_videofile12.mp4'
# orig_file = '/home/flipajs/my_video-16.mkv'
orig_file = '/media/flipajs/Seagate Expansion Drive/bigLenses_colormarks1.avi'
# orig_file = '/home/flipajs/my_video-16.mkv'

# vid_file = '/home/flipajs/Dropbox/PycharmProjects/ants/scripts/my_output_videofile12.mp4'
# vid_file = '/home/flipajs/my_video-16-new3.avi'
# vid_file = '/home/flipajs/my_video-16-DIVX.avi'
# vid_file = '/media/flipajs/Seagate Expansion Drive/smallLense_colony1.avi'
vid_file = '/media/flipajs/Seagate Expansion Drive/blc1_15g1.mkv'
# vid_file = '/home/flipajs/out5_20.mkv'

vid1 = video_manager.VideoManager(orig_file)
vid2 = video_manager.VideoManager(vid_file)

# vid2.seek_frame2(511)
frame = 0
while True:
    img = vid1.next_img()
    if img is None:
        break

    if frame % 1 == 0:
    # if frame > 510:
        # cv2.imshow("fbf", img)

        img2 = vid2.seek_frame2(frame)
        # print frame, vid1.capture.get(cv2.CAP_PROP_POS_MSEC), vid2.capture.get(cv2.CAP_PROP_POS_MSEC)
        # cv2.imshow("seek", img2)

        res = img-img2
        s = np.sum(np.sum(np.sum(res)))
        m = np.amax(res)

        print s, s/(res.shape[0]*res.shape[1]*3), m

        if s != 0:
            print "PROBLEM ", frame

            cv2.imwrite('/home/flipajs/tempimgs2/'+str(frame)+'_o.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            cv2.imwrite('/home/flipajs/tempimgs2/'+str(frame)+'_c.png', img2, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # cv2.waitKey(0)
    frame += 1