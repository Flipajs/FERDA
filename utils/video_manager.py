__author__ = 'filip@naiser.cz'

import cv2
import cv_compatibility
from random import randint
import numpy as np


class VideoType:
    """Enum type to easily distinguish between ordinary video (one video file) and Ferda compressed, which is
    composed from 2 files. Highly compressed full video and segmented second one.
    """
    ORDINARY = 1
    FERDA_COMPRESSED = 2

    def __init__(self):
        pass


class VideoManager():
    """
    this class encapsulates video capturing using OpenCV class VideoCapture
    """

    def __init__(self, video_path, start_t=0, end_t=np.inf, buffer_length=51, crop_model=None):
        """
        :type video_path: str,
        :type buffer_length: int, determines internal buffer length, which allows going back into history without
        frame seeking.
        """

        self.capture = cv2.VideoCapture(video_path)  # OpenCV video capture class
        self.start_t = start_t if start_t > 0 else 0
        self.end_t = end_t if start_t < end_t <= self.video_frame_count_without_bounds() else np.inf

        self.video_path = video_path
        self.buffer_position_ = 0
        self.buffer_length_ = buffer_length  #
        self.view_position_ = self.buffer_length_ - 1  #
        self.buffer_ = [None] * self.buffer_length_
        self.position_ = -1
        self.crop_model = crop_model

        if not self.capture.isOpened():
            raise Exception("Cannot open video! Path: " + video_path)

        self.init_video()

    def inc_pos_(self, pos, volume=1):
        if pos + volume > self.buffer_length_ - 1:
            pos = 0
        else:
            pos += volume

        return pos

    def dec_pos_(self, pos, volume=1):
        if pos - volume < 0:
            pos = self.buffer_length_ - 1
        else:
            pos -= volume

        return pos

    def crop_(self, img):
        cm = self.crop_model
        return img[cm['y1']:cm['y2'], cm['x1']:cm['x2']].copy()

    def next_frame(self):
        # continue reading new frames
        if self.dec_pos_(self.buffer_position_) == self.view_position_:
            f, self.buffer_[self.buffer_position_] = self.capture.read()
            if self.crop_model:
                self.buffer_[self.buffer_position_] = self.crop_(self.buffer_[self.buffer_position_])

            if not f or self.position_ >= self.total_frame_count():
                print "No more frames, end of video file. (video_manager.py)"
                return None

            self.buffer_position_ = self.inc_pos_(self.buffer_position_)

        self.view_position_ = self.inc_pos_(self.view_position_)
        self.position_ += 1
        return self.buffer_[self.view_position_]

    def prev_frame(self):
        return self.previous_frame()

    def previous_frame(self):
        if self.position_ > 0:
            self.position_ -= 1
            view_dec = self.dec_pos_(self.view_position_)
            if (view_dec == self.buffer_position_) or (self.buffer_[view_dec] is None):
                self.buffer_position_ = self.dec_pos_(self.buffer_position_)
                self.view_position_ = self.dec_pos_(self.view_position_)
                self.buffer_[self.view_position_] = self.seek_frame(self.position_)

                # if self.crop_model:
                #     self.buffer_[self.view_position_] = self.crop_(self.buffer_[self.view_position_])

                return self.buffer_[self.view_position_]
            else:
                self.view_position_ = view_dec
                return self.buffer_[self.view_position_]

    def init_video(self):
        if self.start_t > 0:
            # that is enough... when next_frame is called, the position will be processed
            self.capture.set(cv_compatibility.cv_CAP_PROP_POS_FRAMES, self.start_t)

    def seek_frame(self, frame_number):
        if frame_number < 0 or frame_number >= self.total_frame_count():
            frame_number = self.total_frame_count() - 1
            # raise Exception("Frame_number is invalid: "+str(frame_number))

        frame_number_ = frame_number + self.start_t
        # Reset buffer as buffered images are now from other part of the video
        self.buffer_position_ = 0
        self.view_position_ = self.buffer_length_ - 1
        self.buffer_ = [None] * self.buffer_length_

        self.capture.set(cv_compatibility.cv_CAP_PROP_POS_FRAMES, frame_number_)

        # because in next_frame it will be increased by one as usual
        self.position_ = frame_number - 1

        return self.next_frame()

    def img(self):
        return self.buffer_[self.view_position_]

    def random_frame(self):
        """
        gives completely random frame from video
        :return:
        """
        frame_num = self.total_frame_count()
        random_f = randint(0, frame_num)

        return self.seek_frame(random_f)

    def frame_number(self):
        return self.position_
        # return self.capture.get(cv_compatibility.cv_CAP_PROP_POS_FRAMES)

    def fps(self):
        return self.capture.get(cv_compatibility.cv_CAP_PROP_FPS)

    def total_frame_count(self):
        vid_frame_num = int(self.capture.get(cv_compatibility.cv_CAP_PROP_FRAME_COUNT))
        if self.end_t < np.inf:
            vid_frame_num -= vid_frame_num - self.end_t

        if self.start_t > 0:
            vid_frame_num -= self.start_t

        return vid_frame_num

    def video_frame_count_without_bounds(self):
        return int(self.capture.get(cv_compatibility.cv_CAP_PROP_FRAME_COUNT))

    def reset(self):
        """
        resets buffer and also OpenCV VideoCapture, so the video file is being processed from beginning
        """
        self.capture = cv2.VideoCapture(self.video_path)
        self.buffer_position_ = 0
        self.buffer_length_ = 51
        self.view_position_ = self.buffer_length_ - 1
        self.reset_buffer()
        self.position_ = -1

    def reset_buffer(self):
        self.buffer_ = [None] * self.buffer_length_

    def get_manager_copy(self):
        """
        returns copy of VideoManager, might be useful in cases of asynchronous operations (mainly seeking) on video
        while you want to maintain right position in original one.
        """
        vid = VideoManager(self.video_path, start_t=self.start_t, end_t=self.end_t)
        vid.seek_frame(self.frame_number())

        return vid

    def get_frame(self, frame):
        """
        If you ask for given frame, it will access it sequentially or by random access, depending on what is "cheaper"
        With sequence_access = auto = False it behaves the same as frame_seek
        """

        sequence_access = False

        if abs(frame - self.frame_number()) < 15:
            sequence_access = True

        reversed = False
        if frame < self.frame_number():
            reversed = True

        if sequence_access:
            if reversed:
                while self.frame_number() > frame:
                    if self.previous_frame() is None:
                        return None
            else:
                while self.frame_number() < frame:
                    if self.next_frame() is None:
                        return None

            return self.img()
        else:
            return self.seek_frame(frame)


def get_auto_video_manager(project):
    """
    based on file_paths return VideoManager or FerdaCompressedVideoManager instance
    :type file_paths: [str]
    """

    file_paths = project.video_paths
    crop_model = None
    try:
        crop_model = project.video_crop_model
    except:
        pass

    if isinstance(file_paths, list) and len(file_paths) == 1:
        file_paths = file_paths[0]

    if not isinstance(file_paths, list):
        return VideoManager(file_paths, start_t=project.video_start_t, end_t=project.video_end_t, crop_model=crop_model)

    # test which one is the lossless one
    compressed = file_paths[0]
    lossless = file_paths[1]

    c_v = VideoManager(compressed, start_t=project.video_start_t, end_t=project.video_end_t)
    c_img = c_v.next_frame()

    l_v = VideoManager(lossless, start_t=project.video_start_t, end_t=project.video_end_t)
    l_img = l_v.next_frame()

    c_mask = (c_img[:, :, 0] == 255) & (c_img[:, :, 1] == 255) & (c_img[:, :, 2] == 255)
    l_mask = (l_img[:, :, 0] == 255) & (l_img[:, :, 1] == 255) & (l_img[:, :, 2] == 255)

    if sum(sum(c_mask)) > sum(sum(l_mask)):
        compressed = lossless
        lossless = file_paths[0]

    from utils.ferda_compressed_video_manager import FerdaCompressedVideoManager
    return FerdaCompressedVideoManager(compressed, lossless, start_t=project.video_start_t, end_t=project.video_end_t)


def optimize_frame_access_vertices(vertices, project, ra_n_times_slower=40):
    mapping = {}
    regions = []
    for v in vertices:
        r = project.gm.region(v)
        mapping[r] = v
        regions.append(r)

    results = optimize_frame_access(regions, ra_n_times_slower)

    new_results = []
    for r, b1, b2 in results:
        v = mapping[r]
        new_results.append((v, b1, b2))

    return new_results


def optimize_frame_access(list_data, ra_n_times_slower=40):
    """
    implemented by Simon Mandlik

    list_data should be array of nodes, must have .frame_ var
    returns list of tuples in format (data, ra_access [bool], stay_on_same_frame [bool])
    :param list_data:
    :param ra_n_times_slower:
    :return:
    """

    list_data = list(list_data)
    sorted_list = sorted(list_data, key=lambda x: x.frame_)
    result = []
    prev_frame = 0

    while sorted_list:
        node = sorted_list.pop(0)
        frame = node.frame_
        prev_bool = frame == prev_frame
        if (frame - prev_frame) <= ra_n_times_slower:
            tup = (node, True, prev_bool)
            result.append(tup)
        else:
            tup = (node, False, prev_bool)
            result.append(tup)
        prev_frame = frame

    return result


if __name__ == '__main__':
    from core.project.project import Project
    p = Project()
    p.load('/Users/flipajs/Documents/wd/video_bounds_test/test.fproj')

    p.video_start_t = -1
    p.video_end_t = -1

    vid = get_auto_video_manager(p)
