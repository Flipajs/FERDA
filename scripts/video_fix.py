__author__ = 'flipajs'

import cv2

path = '/home/flipajs/'
name = 'my_video-16'
ext = '.mkv'
capture = cv2.VideoCapture(path+name+ext)


f, img = capture.read()
width = img.shape[1]
height = img.shape[0]

vid_writer = cv2.VideoWriter(filename=path+name+"-new2.avi",  #Provide a file to write the video to
                                # fourcc=cv2.VideoWriter_fourcc('i','Y', 'U', ' V'),            #Use whichever codec works for you...
                                # fourcc=cv2.VideoWriter_fourcc('M','J','P','G'),
                                fourcc=cv2.VideoWriter_fourcc('H','2','6','4'),
                                fps=30,                                        #How many frames do you want to display per second in your video?
                                frameSize=(width, height))

frame = 1
while True:
    if not f:
        print "END of video"
        break

    vid_writer.write(img)

    f, img = capture.read()

    frame += 1

    if frame % 100 == 0:
        print frame
