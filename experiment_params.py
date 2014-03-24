__author__ = 'flip'

import cv2
import my_utils
import pickle


class Params():
    def __init__(self):
        self.border = 0
        self.avg_ant_area = 120
        self.avg_ant_axis_a = 0
        self.avg_ant_axis_b = 0
        self.avg_ant_axis_ratio = 4.2
        self.max_axis_ratio_diff = 2
        self.max_area_diff = 0.6

        #TODO> WTF factor ~ INF
        self.undefined_threshold = 0.000001

        self.mser_times = 0
        self.frame = 0

        self.use_gt = True
        self.fast_start = True

        self.intensity_threshold = 200
        self.dynamic_intensity_threshold = True
        self.dynamic_intensity_threshold_history = 5
        self.dynamic_intensity_threshold_increase = 0.0

        self.show_mser_collection = False
        self.show_ants_collection = True
        self.imshow_decreasing_factor = 0.5
        self.print_mser_info = True
        self.print_matching = True

        self.ab_area_xstart = 0.2
        self.ab_area_ystart = 0.2
        self.ab_area_xmax = 41
        self.ab_area_ymax = 26
        self.ab_area_step = 0.05
        self.ab_area_max = 43.0

        afile = open('data/ab_area_hist_blurred.pkl', 'rb')
        self.ab_area_hist = pickle.load(afile)
        afile.close()

        #self.ant_number = 15
        #self.arena = my_utils.RotatedRect(my_utils.Point(405+self.border, 386+self.border), my_utils.Size(766, 766), 0)
        #self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/NoPlasterNoLid800/NoPlasterNoLid800.m4v"
        #self.predefined_vals = 'NoPlasterNoLid800'
        #self.gt_path = '../data/NoPlasterNoLid800/fixed_out.txt'

        self.ant_number = 8
        self.arena = my_utils.RotatedRect(my_utils.Point(593+self.border, 570+self.border), my_utils.Size(344*2, 344*2), 0)
        self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/eight/eight.m4v"
        self.predefined_vals = 'eight'
        self.gt_path = '../data/eight/fixed_out.txt'

        #self.ant_number = 11
        #self.arena = utils.RotatedRect(utils.Point(665+self.border, 504+self.border), utils.Size(491*2, 491*2), 0)
        #self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/Camera 2.m4v"
        #self.predefined_vals = 'Camera2'

        #self.ant_number = 8
        #self.arena = utils.RotatedRect(utils.Point(593+self.border, 570+self.border), utils.Size(344*2, 344*2), 0)
        #self.video_file_name = ''
        #self.predefined_vals = -1