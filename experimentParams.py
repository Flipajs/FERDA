__author__ = 'flip'

import cv2
import utils


class Params():
    def __init__(self):
        self.border = 0
        #self.arena = utils.RotatedRect(utils.Point(405+self.border, 386+self.border), utils.Size(763, 763), 0)
        self.avg_ant_area = 120
        self.avg_ant_axis_ratio = 4.2
        self.max_axis_ratio_diff = 1
        self.max_area_diff = 0.5

        self.show_mser_collections = True
        #self.ant_number = 15
        #self.arena = utils.RotatedRect(utils.Point(405+self.border, 386+self.border), utils.Size(763, 763), 0)
        #self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/ants/NoPlasterNoLid800.avi"
        #self.predefined_vals = 1

        self.ant_number = 8
        self.arena = utils.RotatedRect(utils.Point(593+self.border, 570+self.border), utils.Size(344*2, 344*2), 0)
        self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/eight/eight.avi"
        self.predefined_vals = 2

        #self.ant_number = 8
        #self.arena = utils.RotatedRect(utils.Point(593+self.border, 570+self.border), utils.Size(344*2, 344*2), 0)
        #self.video_file_name = ''
        #self.predefined_vals = -1

    def set_arena(self, arena):
        self.arena = arena

    def arena(self):
        self.arena