from __future__ import print_function
__author__ = 'filip@naiser.cz'

__author__ = 'flip'

import cv2
import my_utils

class VideoManager():
    def __init__(self, video_path):
        self.img = None
        self.capture = cv2.VideoCapture(video_path)
        self.buffer_pos = 0
        self.buffer_length = 51
        self.view_pos = self.buffer_length-1
        self.buffer = [None]*self.buffer_length

        if not self.capture.isOpened():
            print("Cannot open video! video_manager.py")

    def inc_pos(self, pos, volume=1):
        if pos + volume > self.buffer_length - 1:
            pos = 0
        else:
            pos += volume

        return pos

    def dec_pos(self, pos, volume=1):
        if pos - volume < 0:
            pos = self.buffer_length - 1
        else:
            pos -= volume

        return pos

    def next_img(self):
        print("NEXT")
        print("V", self.view_pos)
        print("B", self.buffer_pos)
        #continue reading new frames
        print(self.dec_pos(self.buffer_pos))
        if self.dec_pos(self.buffer_pos) == self.view_pos:
            print(" test")
            f, self.buffer[self.buffer_pos] = self.capture.read()
            if not f:
                print("No more frames > video_manager.py")
                return None

            self.buffer_pos = self.inc_pos(self.buffer_pos)

        self.view_pos = self.inc_pos(self.view_pos)

        return self.buffer[self.view_pos]

    def prev_img(self):
        print("PREV")
        print("V", self.view_pos)
        print("B", self.buffer_pos)
        view_dec = self.dec_pos(self.view_pos)
        if view_dec == self.buffer_pos:
            print("No more frames in buffer")
            return None
        elif self.buffer[view_dec] is None:
            print("No more frames in buffer")
            return None
        else:
            self.view_pos = view_dec
            return self.buffer[self.view_pos]

    def img(self):
        return self.buffer[self.view_pos]


path = "../../data/eight/eight.m4v"
lc = VideoManager(path)

k = 0
while True:
    if k % 128 == 81:
        img = lc.prev_img()
    else:
        img = lc.next_img()

    if img is not None:
        my_utils.imshow("test", img, 0.5)
    else:
        print("ERROR")

    k = cv2.waitKey(0)