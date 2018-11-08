from __future__ import division
from __future__ import unicode_literals
from builtins import range
from builtins import object
from past.utils import old_div
__author__ = 'fnaiser'
from math import floor
import cv2
import numpy as np
from core.bg_model.bg_model import BGModel
from utils import video_manager


class MedianIntensity(object):
    def __init__(self, project, iterations=20, random_frames=False, update_callback=None):
        super(MedianIntensity, self).__init__()
        self.video = video_manager.get_auto_video_manager(project)
        self.iterations = iterations
        self.random_frames = random_frames
        self.update_callback = update_callback
        self.step = -1

    def compute_model(self):
        frame_num = self.video.total_frame_count()
        im = self.video.next_frame()
        imgs = [np.zeros(im.shape, dtype=np.uint8) for i in range(self.iterations)]

        step = int(floor(old_div(frame_num, float(self.iterations))))
        frame_i = 0
        for i in range(self.iterations):
            if self.random_frames:
                im, _ = self.video.random_frame()
            else:
                im = self.video.seek_frame(frame_i)

            imgs[i] = im
            # self.emit(QtCore.SIGNAL('update(int)'), int(100*(i+1)/float(self.iterations)))
            # self.call_update_callback(i)
            self.step = i
            frame_i += step

        self.bg_model = np.asarray(np.percentile(np.array(imgs), 80, axis=0), dtype=np.uint8)
        self.model_ready = True

    def call_update_callback(self, i):
        if self.update_callback:
            self.update_callback(int(100*(i+1)/float(self.iterations)))

    def is_computed(self):
        return self.model_ready

    def get_progress(self):
        return int(100*(self.step+1)/float(self.iterations))

    def get_model(self):
        if self.model_ready:
            m = BGModel(self.bg_model)
            return m

    def update(self, img):
        self.bg_model = np.copy(img)

    def get_fg_mask(self, img, threshold=40):
        return np.logical_or.reduce((self.bg_model.astype(np.int) - img) > threshold, axis=2)


if __name__ == '__main__':
    from core.project.project import Project

    p = Project()
    # p.video_paths = ['/Volumes/Seagate Expansion Drive/IST - videos/colonies/Camera 1.avi']
    p.video_paths = ['/Users/flipajs/Downloads/crickets-out2/out2.mp4']
    # p.video_paths = ['/home/matej/prace/ferda/camera1_ss00-05-00_t00-05-00.mp4']


    num_steps = 10
    bg = MedianIntensity(p, iterations=num_steps)
    bg.compute_model()

    # mi = MaxIntensity(p, iterations=num_steps)
    # mi.compute_model()

    # cv2.imshow('bg max intensity', mi.bg_model)

    # cv2.waitKey(0)

    from utils.video_manager import get_auto_video_manager
    vid = get_auto_video_manager(p)
    import scipy
    model = bg.bg_model
    img = model
    # img = scipy.ndimage.gaussian_filter(model, sigma=5)
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.imshow(model)

    # from PyQt4 import QtGui, QtCore
    import sys
    # app = QtGui.QApplication(sys.argv)
    from gui.view.mser_tree import MSERTree

    plt.figure(2)
    while True:
        im, _ = vid.random_frame()
        # cv2.imshow('orig', im)
        processed = np.subtract(0.8 * np.asarray(bg.bg_model, dtype=np.int32), np.asarray(im, dtype=np.int32))

        processed = (processed - np.min(processed))
        processed /= np.max(processed)

        plt.imshow(processed)
        plt.show()
        # cv2.imshow('sub', processed)
        cv2.waitKey(0)

        processed += -np.min(processed)
        # print np.min(processed), np.max(processed)

        processed[processed < 0] = 0
        processed[processed > 255] = 255
        processed = np.asarray(processed, dtype=np.uint8)
        processed = np.invert(processed)

        fg_mask = np.logical_or.reduce((model.astype(np.int) - im) > 40, axis=2)
        cv2.imshow('foreground mask', fg_mask.astype(np.uint8) * 255)
        im_foreground = np.copy(im)
        im_foreground[~fg_mask] = 0
        cv2.imshow('foreground', im_foreground)

        # ex = MSERTree(processed, p)
        # ex.show()
        # ex.move(-500, -500)
        # ex.showMaximized()
        # ex.setFocus()

        # cv2.imshow('sub', processed)
        cv2.waitKey(0)

    app.exec_()
    app.deleteLater()
    sys.exit()