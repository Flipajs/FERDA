__author__ = 'flip'

import cv2


class VideoManager():
    def __init__(self, video_path):
        self.img = None
        self.capture = cv2.VideoCapture(video_path)
        self.buffer_pos = 0
        self.buffer_length = 51
        self.view_pos = self.buffer_length-1
        self.buffer = [None]*self.buffer_length

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

        return self.buffer[self.view_pos]

    def prev_img(self):
        view_dec = self.dec_pos(self.view_pos)
        if view_dec == self.buffer_pos:
            print "No more frames in buffer"
            return None
        elif self.buffer[view_dec] is None:
            print "No more frames in buffer"
            return None
        else:
            self.view_pos = view_dec
            return self.buffer[self.view_pos]

    def get_prev_img(self):
        view_dec = self.dec_pos(self.view_pos)
        if view_dec == self.buffer_pos:
            print "No more frames in buffer"
            return None
        elif self.buffer[view_dec] is None:
            print "No more frames in buffer"
            return None
        else:
            ret = self.buffer[view_dec]
            return ret

    def seek_frame(self, frame_number):
        self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_number)

        f, img = self.capture.read()
        if not f:
            print "Problem seeking for frame in video_manager.py"
            return None

        return img

    def img(self):
        return self.buffer[self.view_pos]