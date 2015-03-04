__author__ = 'fnaiser'

from model import Model
from math import floor
import numpy as np
from utils import video_manager

class MaxIntensity(Model):
    def __init__(self, video_paths, iterations=10, random_frames=False, update_callback=None):
        super(MaxIntensity, self).__init__()
        self.video = video_manager.get_auto_video_manager(video_paths)
        self.bg_model = None
        self.iterations = iterations
        self.random_frames = random_frames
        self.update_callback = update_callback
        self.step = -1

    def iteration(self, im):
        self.bg_model = np.maximum(self.bg_model, im)

    def compute_model(self):
        frame_num = self.video.total_frame_count()
        im = self.video.move2_next()
        self.bg_model = np.zeros(im.shape, dtype=np.uint8)

        step = int(floor(frame_num / float(self.iterations)))
        frame_i = 0
        for i in range(self.iterations):
            if self.random_frames:
                im = self.video.random_frame()
            else:
                im = self.video.seek_frame(frame_i)

            self.iteration(im)
            self.call_update_callback(i)
            self.step = i
            frame_i += step

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
            return self.bg_model