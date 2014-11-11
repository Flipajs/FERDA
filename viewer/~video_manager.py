__author__ = 'flip'

import cv2
import cv_compatibility


class VideoManager():


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

    def move2_next(self):
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

    def move2_prev(self):
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
        self.capture.set(cv_compatibility.cv_CAP_PROP_POS_FRAMES, frame_number)
        while pos < frame_number:
            pos = self.capture.get(cv_compatibility.cv_CAP_PROP_POS_FRAMES)
            ret, image = self.capture.read()
            if pos == frame_number:
                return image
            elif pos > frame_number:
                position_to_set -= 1
                self.capture.set(cv_compatibility.cv_CAP_PROP_POS_FRAMES, position_to_set)
                pos = -1

    def img(self):
        return self.buffer[self.view_pos]

    def frame_number(self):
        return self.position





