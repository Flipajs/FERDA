__author__ = 'flipajs'

import cv2
import numpy as np
from viewer import video_manager

# vid_file = '/home/flipajs/Dropbox/PycharmProjects/ants/scripts/my_output_videofile12.mp4'
# orig_file = '/home/flipajs/my_video-16.mkv'
orig_file = '/media/flipajs/Seagate Expansion Drive/IST - videos/tests/output.avi'
orig_file = '/home/flipajs/my_video-16.mkv'

# vid_file = '/home/flipajs/Dropbox/PycharmProjects/ants/scripts/my_output_videofile12.mp4'
# vid_file = '/home/flipajs/my_video-16-new3.avi'
# vid_file = '/home/flipajs/my_video-16-DIVX.avi'
# vid_file = '/media/flipajs/Seagate Expansion Drive/smallLense_colony1.avi'
vid_file10 = '/media/flipajs/Seagate Expansion Drive/IST - videos/tests/out_c16_gf30.avi'
vid_file18 = '/media/flipajs/Seagate Expansion Drive/IST - videos/tests/out_c18_gfdef.mkv'
vid_file20 = '/media/flipajs/Seagate Expansion Drive/IST - videos/tests/out_c20_gf30.avi'

vid_file18 = '/home/flipajs/my_video-16.mkv'
# vid_file18 = '/home/flipajs/my_video-16_c18_gf30.avi'
# vid_file = '/home/flipajs/out5_20.mkv'

vid1 = video_manager.VideoManager(orig_file)
vid18 = video_manager.VideoManager(vid_file18)

vid18_r = video_manager.VideoManager(vid_file18)

# vid2.seek_frame2(511)
frame = 0
print "compression", "err_sum", "err_mean", "err_max", "err_med"
while True:
    # img = vid1.next_img()
    # if img is None:
    #     break

    img10_r = vid18_r.next_img()

    if frame % 13 == 0 or frame % 27 == 0:
        print frame

        vid10 = video_manager.VideoManager(vid_file18)
        img10 = vid10.seek_frame_hybrid(frame)



        # res10 = np.absolute(np.asarray(img10, dtype=np.int32) - np.asarray(img, dtype=np.int32))

        # s10 = np.sum(np.sum(np.sum(res10)))
        # m10 = np.amax(res10)
        # me10 = np.median(res10)

        res10_r = np.sum(np.sum(np.sum(np.absolute(np.asarray(img10, dtype=np.int32) - np.asarray(img10_r, dtype=np.int32)))))
        # print "18: ", s10, s10 / float(img.shape[0]*img.shape[1]*3), m10, me10

        if res10_r > 0:
            print "PROBLEM ", frame


        # cv2.imwrite('/home/flipajs/tempimgs3/'+str(frame)+'_o.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # cv2.imwrite('/home/flipajs/tempimgs3/'+str(frame)+'_18.png', img10, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


    frame += 1


# __author__ = 'flipajs'
#
# import cv2
# import numpy as np
# from viewer import video_manager
#
# # vid_file = '/home/flipajs/Dropbox/PycharmProjects/ants/scripts/my_output_videofile12.mp4'
# # orig_file = '/home/flipajs/my_video-16.mkv'
# orig_file = '/media/flipajs/Seagate Expansion Drive/IST - videos/tests/output.avi'
# # orig_file = '/home/flipajs/my_video-16.mkv'
#
# # vid_file = '/home/flipajs/Dropbox/PycharmProjects/ants/scripts/my_output_videofile12.mp4'
# # vid_file = '/media/flipajs/Seagate Expansion Drive/smallLense_colony1.avi'
# # vid_file = '/home/flipajs/my_video-16-new3.avi'
# # vid_file = '/home/flipajs/my_video-16-DIVX.avi'
# vid_file10 = '/media/flipajs/Seagate Expansion Drive/IST - videos/tests/out_c16_gf30.avi'
# vid_file15 = '/media/flipajs/Seagate Expansion Drive/IST - videos/tests/out_c18_gf30.avi'
# vid_file20 = '/media/flipajs/Seagate Expansion Drive/IST - videos/tests/out_c20_gf30.avi'
#
# # vid_file = '/home/flipajs/out5_20.mkv'
#
# vid1 = video_manager.VideoManager(orig_file)
# vid10 = video_manager.VideoManager(vid_file10)
# vid15 = video_manager.VideoManager(vid_file15)
# vid20 = video_manager.VideoManager(vid_file20)
#
# vid10_r = video_manager.VideoManager(vid_file10)
# vid15_r = video_manager.VideoManager(vid_file15)
# vid20_r = video_manager.VideoManager(vid_file20)
#
# # vid2.seek_frame2(511)
# frame = 0
# print "compression", "err_sum", "err_mean", "err_max", "err_med"
# while True:
#     img = vid1.next_img()
#     if img is None:
#         break
#     print
#     # print "FRAME: ", frame
#     if frame % 1 == 0:
#     # if frame > 510:
#         # cv2.imshow("fbf", img)
#
#         img10 = vid10.seek_frame_hybrid(frame)
#         img15 = vid15.seek_frame_hybrid(frame)
#         img20 = vid20.seek_frame_hybrid(frame)
#
#         img10_r = vid10_r.next_img()
#         img15_r = vid15_r.next_img()
#         img20_r = vid20_r.next_img()
#
#         # res = img[:,:,0]-img2[:,:,1]
#         res10 = np.absolute(np.asarray(img10, dtype=np.int32) - np.asarray(img, dtype=np.int32))
#         res15 = np.absolute(np.asarray(img15, dtype=np.int32) - np.asarray(img, dtype=np.int32))
#         res20 = np.absolute(np.asarray(img20, dtype=np.int32) - np.asarray(img, dtype=np.int32))
#
#         s10 = np.sum(np.sum(np.sum(res10)))
#         m10 = np.amax(res10)
#         me10 = np.median(res10)
#
#         s15 = np.sum(np.sum(np.sum(res15)))
#         m15 = np.amax(res15)
#         me15 = np.median(res10)
#
#         s20 = np.sum(np.sum(np.sum(res20)))
#         m20 = np.amax(res20)
#         me20 = np.median(res20)
#
#         # print "10: ", s10, s10 / float(img.shape[0]*img.shape[1]*3), m10, me10
#         # print "15: ", s15, s15 / float(img.shape[0]*img.shape[1]*3), m15, me15
#         # print "20: ", s20, s20 / float(img.shape[0]*img.shape[1]*3), m20, me20
#
#         res10_r = np.sum(np.sum(np.sum(np.absolute(np.asarray(img10, dtype=np.int32) - np.asarray(img10_r, dtype=np.int32)))))
#         res15_r = np.sum(np.sum(np.sum(np.absolute(np.asarray(img15, dtype=np.int32) - np.asarray(img15_r, dtype=np.int32)))))
#         res20_r = np.sum(np.sum(np.sum(np.absolute(np.asarray(img20, dtype=np.int32) - np.asarray(img20_r, dtype=np.int32)))))
#
#         if res10_r > 0 or res15_r > 0 or res20_r > 0:
#             print "PROBLEM ", frame
#
#         # if s != 0:
#         #     print "PROBLEM ", frame
#
#         # cv2.imwrite('/home/flipajs/tempimgs2/'+str(frame)+'_d.png', np.asarray(np.absolute(res), dtype=np.uint8), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
#         cv2.imwrite('/home/flipajs/tempimgs2/'+str(frame)+'_o.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
#         cv2.imwrite('/home/flipajs/tempimgs2/'+str(frame)+'_10.png', img10, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
#         cv2.imwrite('/home/flipajs/tempimgs2/'+str(frame)+'_15.png', img15, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
#         cv2.imwrite('/home/flipajs/tempimgs2/'+str(frame)+'_20.png', img20, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
#         # cv2.waitKey(0)
#     frame += 1