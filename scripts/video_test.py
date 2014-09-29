__author__ = 'flipajs'

import cv2
from viewer import video_manager

# vid_file = '/home/flipajs/Dropbox/PycharmProjects/ants/scripts/my_output_videofile12.mp4'
orig_file = '/home/flipajs/my_video-16.mkv'
# vid_file = '/home/flipajs/Dropbox/PycharmProjects/ants/scripts/my_output_videofile12.mp4'
# vid_file = '/home/flipajs/my_video-16-new3.avi'
vid_file = '/home/flipajs/my_video-16-DIVX.avi'
vid1 = video_manager.VideoManager(orig_file)
vid2 = video_manager.VideoManager(vid_file)

frame = 0
while True:
    img = vid1.next_img()
    if img is None:
        break

    if frame % 100 == 0:
        print frame
        cv2.imshow("fbf", img)

        img2 = vid2.seek_frame(frame)
        cv2.imshow("seek", img2)
        cv2.waitKey(0)


    frame += 1