__author__ = 'flip'

import cv2
import os
import numpy as np


class VideoManager():
    def __init__(self, video_path):
        self.img = None
        self.capture = cv2.VideoCapture(video_path)
        self.video_path = video_path
        self.buffer_pos = 0
        self.buffer_length = 51
        self.view_pos = self.buffer_length-1
        self.buffer = [None]*self.buffer_length
        self.position = -1

        if not self.capture.isOpened():
            print "Cannot open video! video_manager.py"

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
        #continue reading new frames
        #print self.dec_pos(self.buffer_pos)
        if self.dec_pos(self.buffer_pos) == self.view_pos:
            f, self.buffer[self.buffer_pos] = self.capture.read()
            if not f:
                print "No more frames > video_manager.py"
                return None

            self.buffer_pos = self.inc_pos(self.buffer_pos)

        self.view_pos = self.inc_pos(self.view_pos)
        self.position += 1
        return self.buffer[self.view_pos]

    def prev_img(self):
        if self.position > 0:
            self.position -= 1
            view_dec = self.dec_pos(self.view_pos)
            if (view_dec == self.buffer_pos) or (self.buffer[view_dec] is None):
                self.buffer_pos = self.dec_pos(self.buffer_pos)
                self.view_pos = self.dec_pos(self.view_pos)
                self.buffer[self.view_pos] = self.seek_frame(self.position)
                return self.buffer[self.view_pos]
            else:
                self.view_pos = view_dec
                return self.buffer[self.view_pos]

    # def get_prev_img(self):
    # 	view_dec = self.dec_pos(self.view_pos)
    # 	if view_dec == self.buffer_pos:
    # 		print "No more frames in buffer"
    # 		return None
    # 	elif self.buffer[view_dec] is None:
    # 		print "No more frames in buffer"
    # 		return None
    # 	else:
    # 		ret = self.buffer[view_dec]
    # 		return ret

    def seek_frame(self, frame_number):
        if frame_number < 0 or frame_number >= self.total_frame_count():
            return None

        #Reset buffer as buffered images are now from other part of the video
        self.buffer_pos = 0
        self.view_pos = self.buffer_length-1
        self.buffer = [None]*self.buffer_length

        self.position = frame_number
        position_to_set = frame_number
        pos = -1
        self.capture.set(cv2.cv.CAP_PROP_POS_FRAMES, frame_number)
        while pos < frame_number:
            pos = self.capture.get(cv2.cv.CAP_PROP_POS_FRAMES)
            ret, image = self.capture.read()
            if pos == frame_number:
                return image
            elif pos > frame_number:
                print "DECREASING"
                position_to_set -= 1
                self.capture.set(cv2.cv.CAP_PROP_POS_FRAMES, position_to_set)
                pos = -1

        # Original not-always-working solution:
        # self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_number)
        # self.position = frame_number - 1
        # return self.next_img()

    def get_ms(self, fps, frame):
        full_s = frame / int(fps)
        rest = frame - (full_s * fps)

        ms = full_s * 1000 + (rest * 1000) / fps

        # ms = (frame * 1000) / int(fps)
        return ms

    def seek_frame2(self, frame_number):
        if frame_number < 0 or frame_number >= self.total_frame_count():
            return None

        #Reset buffer as buffered images are now from other part of the video
        self.buffer_pos = 0
        self.view_pos = self.buffer_length-1
        self.buffer = [None]*self.buffer_length

        self.position = frame_number
        f_ms = 1000/self.fps()
        position_to_set = self.get_ms(self.fps(), frame_number)
        ms = self.get_ms(self.fps(), frame_number)
        pos = -1

        self.capture.set(cv2.cv.CAP_PROP_POS_MSEC, ms)
        while pos < ms:
            pos = self.capture.get(cv2.cv.CAP_PROP_POS_MSEC)
            ret, image = self.capture.read()
            if np.allclose(pos, ms):
                return image
            elif pos > ms:
                print "DECREASING"
                position_to_set -= f_ms
                self.capture.set(cv2.cv.CAP_PROP_POS_MSEC, position_to_set)
                pos = -(1000)/self.fps()

        # Original not-always-working solution:
        # self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_number)
        # self.position = frame_number - 1
        # return self.next_img()

    def img(self):
        return self.buffer[self.view_pos]

    def frame_number(self):
        return self.position

    def fps(self):
        return self.capture.get(cv2.cv.CAP_PROP_FPS)

    def total_frame_count(self):
        return self.capture.get(cv2.cv.CAP_PROP_FRAME_COUNT)

    def reset(self):
        self.img = None
        self.capture = cv2.VideoCapture(self.video_path)
        self.buffer_pos = 0
        self.buffer_length = 51
        self.view_pos = self.buffer_length-1
        self.buffer = [None]*self.buffer_length
        self.position = -1