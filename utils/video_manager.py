__author__ = 'filip@naiser.cz'

import cv2
import cv_compatibility
from utils.ferda_compressed_video_manager import FerdaCompressedVideoManager
from random import randint


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

    def __init__(self, video_path, buffer_length=51):
        """
        :type video_path: str,
        :type buffer_length: int, determines internal buffer length, which allows going back into history without
        frame seeking.
        """

        self.capture = cv2.VideoCapture(video_path)  # OpenCV video capture class
        self.video_path = video_path
        self.buffer_position_ = 0
        self.buffer_length_ = buffer_length  #
        self.view_position_ = self.buffer_length_ - 1  #
        self.buffer_ = [None] * self.buffer_length_
        self.position_ = -1

        if not self.capture.isOpened():
            raise Exception("Cannot open video! Path: " + video_path)

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

    def move2_next(self):
        # continue reading new frames
        if self.dec_pos_(self.buffer_position_) == self.view_position_:
            f, self.buffer_[self.buffer_position_] = self.capture.read()
            if not f:
                print "No more frames > video_manager.py"
                return None

            self.buffer_position_ = self.inc_pos_(self.buffer_position_)

        self.view_position_ = self.inc_pos_(self.view_position_)
        self.position_ += 1
        return self.buffer_[self.view_position_]

    def move2_prev(self):
        if self.position_ > 0:
            self.position_ -= 1
            view_dec = self.dec_pos_(self.view_position_)
            if (view_dec == self.buffer_position_) or (self.buffer_[view_dec] is None):
                self.buffer_position_ = self.dec_pos_(self.buffer_position_)
                self.view_position_ = self.dec_pos_(self.view_position_)
                self.buffer_[self.view_position_] = self.seek_frame(self.position_)
                return self.buffer_[self.view_position_]
            else:
                self.view_position_ = view_dec
                return self.buffer_[self.view_position_]

    def get_prev_img(self):
        """
        this method allows to get previous frame from buffer without moving
        actual position.
        """

        view_dec = self.dec_pos_(self.view_position_)
        if view_dec == self.buffer_position_:
            print "No more frames in buffer"
            return None
        elif self.buffer_[view_dec] is None:
            print "No more frames in buffer"
            return None
        else:
            ret = self.buffer_[view_dec]
            return ret

    def seek_frame(self, frame_number):
        if frame_number < 0 or frame_number >= self.total_frame_count():
            raise Exception("Frame_number is invalid: "+str(frame_number))

        # Reset buffer as buffered images are now from other part of the video
        self.buffer_position_ = 0
        self.view_position_ = self.buffer_length_ - 1
        self.buffer_ = [None] * self.buffer_length_

        self.capture.set(cv_compatibility.cv_CAP_PROP_POS_FRAMES, frame_number)

        # because in move2_next it will be increased by one as usual
        self.position_ = frame_number - 1

        return self.move2_next()

        # position_to_set = frame_number
        # pos = -1
        # self.capture.set(cv_compatibility.cv_CAP_PROP_POS_FRAMES, frame_number)
        # while pos < frame_number:
        #     pos = self.capture.get(cv_compatibility.cv_CAP_PROP_POS_FRAMES)
        #     ret, image = self.capture.read()
        #     if pos == frame_number:
        #         return image
        #     elif pos > frame_number:
        #         position_to_set -= 1
        #         self.capture.set(cv_compatibility.cv_CAP_PROP_POS_FRAMES, position_to_set)
        #         pos = -1

    def img(self):
        return self.buffer_[self.view_position_]

    def random_frame(self):
        frame_num = self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        random_f = randint(0, frame_num)

        return self.seek_frame(random_f)

    def frame_number(self):
        return self.position_
        # return self.capture.get(cv_compatibility.cv_CAP_PROP_POS_FRAMES)

    def fps(self):
        return self.capture.get(cv_compatibility.cv_CAP_PROP_FPS)

    def total_frame_count(self):
        return int(self.capture.get(cv_compatibility.cv_CAP_PROP_FRAME_COUNT))

    def reset(self):
        """
        resets buffer and also OpenCV VideoCapture, so the video file is being processed from beginning
        """
        self.capture = cv2.VideoCapture(self.video_path)
        self.buffer_position_ = 0
        self.buffer_length_ = 51
        self.view_position_ = self.buffer_length_ - 1
        self.buffer_ = [None] * self.buffer_length_
        self.position_ = -1


        # def move2_prev(self):
        # view_dec = self.dec_pos(self.view_pos)
        # if view_dec == self.buffer_pos:
        # print "No more frames in buffer"
        # return None
        #     elif self.buffer[view_dec] is None:
        #         print "No more frames in buffer"
        #         return None
        #     else:
        #         self.view_pos = view_dec
        #         return self.buffer[self.view_pos]

    def get_manager_copy(self):
        """
        returns copy of VideoManager, might be useful in cases of asynchronous operations (mainly seeking) on video
        while you want to maintain right position in original one.
        """
        vid = VideoManager(self.video_path)
        vid.seek_frame(self.frame_number())

        return vid

    # TODO: add this and other functions into prototype class...
    def get_frame(self, frame, sequence_access=False, auto=False):
        if auto:
            if abs(frame - self.frame_number()) < 15:
                sequence_access = True

        reversed = False
        if frame < self.frame_number():
            reversed = True

        if sequence_access:
            if reversed:
                while self.frame_number() > frame:
                    self.move2_prev()
            else:
                while self.frame_number() < frame:
                    self.move2_next()

            return self.img()
        else:
            return self.seek_frame(frame)


def get_auto_video_manager(file_paths):
    """
    based on file_paths return VideoManager or FerdaCompressedVideoManager instance
    :type file_paths: [str]
    """

    if not isinstance(file_paths, list):
        return VideoManager(file_paths)

    if len(file_paths) == 1:
        return VideoManager(file_paths[0])
    else:
        # test which one is the lossless one
        compressed = file_paths[0]
        lossless = file_paths[1]

        c_v = VideoManager(compressed)
        c_img = c_v.move2_next()

        l_v = VideoManager(lossless)
        l_img = l_v.move2_next()

        c_mask = (c_img[:, :, 0] == 255) & (c_img[:, :, 1] == 255) & (c_img[:, :, 2] == 255)
        l_mask = (l_img[:, :, 0] == 255) & (l_img[:, :, 1] == 255) & (l_img[:, :, 2] == 255)

        if sum(sum(c_mask)) > sum(sum(l_mask)):
            compressed = lossless
            lossless = file_paths[0]

        return FerdaCompressedVideoManager(compressed, lossless)


def optimize_frame_access(list_data, ra_n_times_slower=40):
    """
    implemented by Simon Mandlik

    list_data should be array of nodes, must have .frame_ var
    returns list of tuples in format (data, ra_access [bool], stay_on_same_frame [bool])
    :param list_data:
    :param ra_n_times_slower:
    :return:
    """

    sorted_list = list(list_data)
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


if __name__ == "__main__":
    # test lossless detection:

    compressed = "/home/flipajs/segmentation/camera1_test_c25.avi"
    lossless = "/home/flipajs/segmentation/out.avi"
    file_paths = [compressed, lossless]

    get_auto_video_manager(file_paths)