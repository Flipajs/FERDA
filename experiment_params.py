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
        self.min_margin = 5

        #TODO> WTF factor ~ INF
        self.undefined_threshold = 0.000001

        self.weighted_matching_lost_edge_cost = 0.0001

        self.mser_times = 0
        self.frame = 0

        self.use_gt = True
        self.gt_repair = False
        self.fast_start = True

        self.intensity_threshold = 100
        self.dynamic_intensity_threshold = True
        self.dynamic_intensity_threshold_history = 5
        self.dynamic_intensity_threshold_increase = 0.0

        self.show_mser_collection = False
        self.show_ants_collection = True
        self.show_image = True
        self.show_assignment_problem = True
        self.imshow_decreasing_factor = 0.5
        self.print_mser_info = False
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

        afile = open('data/a_area_hist_blurred.pkl', 'rb')
        self.a_area_hist = pickle.load(afile)
        afile.close()

        self.a_area_xstart = 0.2
        self.a_area_ystart = 0.2
        self.a_area_xmax = 41
        self.a_area_ymax = 26
        self.a_area_step = 0.05

        #self.ant_number = 15
        #self.arena = my_utils.RotatedRect(my_utils.Point(403+self.border, 387+self.border), my_utils.Size(764, 764), 0)
        #self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/NoPlasterNoLid800/NoPlasterNoLid800.m4v"
        #self.predefined_vals = 'NoPlasterNoLid800'
        #self.gt_path = '../data/NoPlasterNoLid800/fixed_out.txt'

        self.ant_number = 8
        self.arena = my_utils.RotatedRect(my_utils.Point(593+self.border, 570+self.border), my_utils.Size(344*2, 344*2), 0)
        self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/eight/eight.m4v"
        self.predefined_vals = 'eight'
        self.gt_path = '../data/eight/fixed_out.txt'

        #self.ant_number = 11
        #self.arena = my_utils.RotatedRect(my_utils.Point(665+self.border, 504+self.border), my_utils.Size(491*2, 491*2), 0)
        #self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/Camera 2.m4v"
        #self.predefined_vals = 'Camera2'

        #self.ant_number = 8
        #self.arena = utils.RotatedRect(utils.Point(593+self.border, 570+self.border), utils.Size(344*2, 344*2), 0)
        #self.video_file_name = ''
        #self.predefined_vals = -1