__author__ = 'flip'

import cv2

class LifeCycle():
    def __init__(self, video_path):
        self.img = None
        self.capture = cv2.VideoCapture(video_path)

        if not self.capture.isOpened():
            print "Cannot open video! lifeCycle.py"

    def next_img(self):
        f, self.img = self.capture.read()

        if not f:
            print "No more frames > lifeCycle.py"
            return None

        return self.img

    def img(self):
        return self.img