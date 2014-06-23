__author__ = 'flipajs'
import os
import cv2
import math
import numpy as np

frames = range(700, 4201, 700)
print frames

#dir = os.path.expanduser('~/dump/mesors/frame_results/')
#dir_out = os.path.expanduser('~/dump/')
#
#y = 120
#h = 900
#x = 500
#w = 900
#
#rows = 3
#cols = 2
#font= 4
#textH = 50
#font_width = 2
#name='collection_mesors'
#
frames = range(800, 1601, 800)
print frames

dir = os.path.expanduser('~/dump/zebrafish/frame_results/')
dir_out = os.path.expanduser('~/dump/')

y = 0
h = 1078
x = 197
w = 1525

rows = 2
cols = 1
font=4
textH = 50
font_width = 2
name='collection_zebrafish'

#frames = range(300, 1801, 300)
#print frames
#
#dir = os.path.expanduser('~/dump/noplast/frame_results/')
#dir_out = os.path.expanduser('~/dump/')
#
#y = 0
#h = 800
#x = 0
#w = 800
#
#rows = 3
#cols = 2
#font=3
#font_width=2
#textH = 37
#name='collection_noplast'


#frames = range(250, 1501, 250)
#print frames
#
#dir = os.path.expanduser('~/dump/eight/frame_results/')
#dir_out = os.path.expanduser('~/dump/')
#
#y = 205
#h = 730
#x = 225
#w = 740
#
#rows = 3
#cols = 2
#font=3
#font_width=2
#textH = 37
#name='collection_eight'


#frames = range(15, 101, 15)
#print frames
#
#dir = os.path.expanduser('~/dump/drosophyla/frame_results/')
#dir_out = os.path.expanduser('~/dump/')
#
#y = 0
#h = 1024
#x = 0
#w = 1024
#
#rows = 3
#cols = 2
#font=5
#textH = 60
#font_width = 4
#name='collection_drosophyla'


#frames = range(681, 693)
#print frames
#
#dir = os.path.expanduser('~/dump/eight/frames/')
#dir_out = os.path.expanduser('~/dump/')
#
#y = 340
#h = 140
#x = 750
#w = 140
#
#rows = 3
#cols = 4
#font = 1
#textH = 13
#font_width = 1
#name='collection_eight691'

collection = np.zeros((h*rows + rows + 1, w*cols + cols + 1, 3))
border = True

counter = 0
for f in frames:
    img = np.array(cv2.imread(dir+str(f)+'.png'))
    crop = img[y:y+h, x:x+w]
    r = math.floor(counter/cols)
    c = counter % cols

    cv2.putText(crop, str(f), (3, textH), cv2.FONT_HERSHEY_PLAIN, font, (0, 0, 0), font_width, cv2.CV_AA)

    collection[r + 1 + r*h:r + 1 + r*h+h, c + 1 + c*w:c + 1 + c*w+w] = crop
    cv2.imwrite(dir_out+str(f)+'.png', crop)

    counter += 1

cv2.imwrite(dir_out+name+'.png', collection)