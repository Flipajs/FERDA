__author__ = 'flip'

import cv2
import utils


class Params():
    def __init__(self):
        self.border = 0
        #self.arena = utils.RotatedRect(utils.Point(405+self.border, 386+self.border), utils.Size(763, 763), 0)
        self.avg_ant_area = 120
        self.avg_ant_axis_a = 0
        self.avg_ant_axis_b = 0
        self.avg_ant_axis_ratio = 4.2
        self.max_axis_ratio_diff = 2
        self.max_area_diff = 0.6

        self.undefined_threshold = 0.00001

        self.mser_times = 0
        self.frame = 0

        self.intensity_threshold = 200
        self.dynamic_intensity_threshold = True
        self.dynamic_intensity_threshold_history = 5
        self.dynamic_intensity_threshold_increase = 0.0

        self.show_mser_collection = True
        self.show_ants_collection = True
        self.show_assignment_problem = True
        self.ant_number = 15
        self.arena = utils.RotatedRect(utils.Point(405+self.border, 386+self.border), utils.Size(763, 763), 0)
        self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/NoPlasterNoLid800/NoPlasterNoLid800.m4v"
        self.predefined_vals = 'NoPlasterNoLid800'

        #self.ant_number = 8
        #self.arena = utils.RotatedRect(utils.Point(593+self.border, 570+self.border), utils.Size(344*2, 344*2), 0)
        #self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/eight/eight.m4v"
        #self.predefined_vals = 'eight'

        #self.ant_number = 11
        #self.arena = utils.RotatedRect(utils.Point(665+self.border, 504+self.border), utils.Size(491*2, 491*2), 0)
        #self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/Camera 2.m4v"
        #self.predefined_vals = 'Camera2'


        #self.ant_number = 8
        #self.arena = utils.RotatedRect(utils.Point(593+self.border, 570+self.border), utils.Size(344*2, 344*2), 0)
        #self.video_file_name = ''
        #self.predefined_vals = -1

    def set_arena(self, arena):
        self.arena = arena

    def arena(self):
        self.arena