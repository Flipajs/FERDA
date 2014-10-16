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
vid_file10 = '/media/flipajs/Seagate Expansion Drive/blc1_10g1.mkv'
vid_file15 = '/media/flipajs/Seagate Expansion Drive/blc1_15g1.mkv'
vid_file20 = '/media/flipajs/Seagate Expansion Drive/blc1_20g1.mkv'
# vid_file = '/home/flipajs/out5_20.mkv'

vid1 = video_manager.VideoManager(orig_file)
vid10 = video_manager.VideoManager(vid_file10)
vid15 = video_manager.VideoManager(vid_file15)
vid20 = video_manager.VideoManager(vid_file20)

# vid2.seek_frame2(511)
frame = 0
print "compression", "err_sum", "err_mean", "err_max", "err_med"
while True:
    img = vid1.next_img()
    if img is None:
        break
    print
    print "FRAME: ", frame
    if frame % 1 == 0:
    # if frame > 510:
        # cv2.imshow("fbf", img)

        img10 = vid10.seek_frame2(frame)
        img15 = vid15.seek_frame2(frame)
        img20 = vid20.seek_frame2(frame)

        # res = img[:,:,0]-img2[:,:,1]
        res10 = np.absolute(np.asarray(img10, dtype=np.int32) - np.asarray(img, dtype=np.int32))
        res15 = np.absolute(np.asarray(img15, dtype=np.int32) - np.asarray(img, dtype=np.int32))
        res20 = np.absolute(np.asarray(img20, dtype=np.int32) - np.asarray(img, dtype=np.int32))

        s10 = np.sum(np.sum(np.sum(res10)))
        m10 = np.amax(res10)
        me10 = np.median(res10)

        s15 = np.sum(np.sum(np.sum(res15)))
        m15 = np.amax(res15)
        me15 = np.median(res10)

        s20 = np.sum(np.sum(np.sum(res20)))
        m20 = np.amax(res20)
        me20 = np.median(res20)

        print "10: ", s10, s10 / float(img.shape[0]*img.shape[1]*3), m10, me10
        print "15: ", s15, s15 / float(img.shape[0]*img.shape[1]*3), m15, me15
        print "20: ", s20, s20 / float(img.shape[0]*img.shape[1]*3), m20, me20

        # if s != 0:
        #     print "PROBLEM ", frame

        # cv2.imwrite('/home/flipajs/tempimgs2/'+str(frame)+'_d.png', np.asarray(np.absolute(res), dtype=np.uint8), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        cv2.imwrite('/home/flipajs/tempimgs2/'+str(frame)+'_o.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        cv2.imwrite('/home/flipajs/tempimgs2/'+str(frame)+'_10.png', img10, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        cv2.imwrite('/home/flipajs/tempimgs2/'+str(frame)+'_15.png', img15, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        cv2.imwrite('/home/flipajs/tempimgs2/'+str(frame)+'_20.png', img20, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # cv2.waitKey(0)
    frame += 1