__author__ = 'flipajs'

import cv2
import subprocess as sp
import numpy as np
import ffmpeg_writer

# path = '/home/flipajs/'
path = '/media/flipajs/Seagate Expansion Drive/'
name = 'bigLenses_colormarks1'
ext = '.avi'
capture = cv2.VideoCapture(path+name+ext)


f, img = capture.read()
width = img.shape[1]
height = img.shape[0]

file_name = path+name+'-5'+ext



# video_writer = ffmpeg_writer.VideoSink(img.shape)


# video_writer = moviepy.FFMPEG_VideoWriter(file_name, [width, height], 30)
#
vid_writer = cv2.VideoWriter(filename=path+name+"-DIVX.avi",  #Provide a file to write the video to
                                # fourcc=cv2.VideoWriter_fourcc('I', 'Y', 'U', 'V'),            #Use whichever codec works for you...
                                # fourcc=cv2.VideoWriter_fourcc('M','J','P','G'),
                                # fourcc=cv2.VideoWriter_fourcc('H','2','6','4'),
                                # fourcc=cv2.VideoWriter_fourcc('I','2','6','3'),
                                fourcc=cv2.VideoWriter_fourcc('D','I','V','X'),
                                # fourcc=cv2.VideoWriter_fourcc('H','F','Y','U'),
                                fps=30,                                        #How many frames do you want to display per second in your video?
                                frameSize=(width, height))


#
#
# # FFMPEG_BIN = "avconv"
# #
# # command = [ FFMPEG_BIN,
# #         '-y', # (optional) overwrite output file if it exists
# #         '-s', str(width)+'x'+str(height), # size of one frame
# #         '-pix_fmt', 'rgb24',
# #         '-r', '30', # frames per second
# #         '-i', '-', # The imput comes from a pipe
# #         '-an', # Tells FFMPEG not to expect any audio
# #         '-vcodec', 'h264',
# #         'my_output_videofile.mp4' ]
# #
# # print command
# # pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
#
frame = 1
while True:
    if not f:
        print "END of video"
        break

    # video_writer.run(img)
    # video_writer.write_frame(img)
    vid_writer.write(img)
    cv2.imwrite(path+"img_test/"+name+str(frame)+".png", img)
    # pipe.proc.stdin.write(img.tostring())
    #
    # cv2.imshow("TEST", img)
    # cv2.waitKey(0)
    f, img = capture.read()

    if frame % 1000 == 0:
        print frame

    frame += 1


# video_writer.close()