__author__ = 'flip'

import cv2
import my_utils
import pickle
import os
import scipy.ndimage
from utils.misc import is_flipajs_pc


class Params():
    def __init__(self):
        self.border = 0
        self.avg_ant_area = 120
        self.avg_ant_axis_a = 1
        self.avg_ant_axis_b = 1
        self.avg_ant_axis_ratio = 4.2
        self.max_axis_ratio_diff = 2
        self.max_area_diff = 0.6
        #mser min margin
        self.min_margin = 5

        self.allow_frame_seek = True

        self.inverted_image = False

        self.skip_big_regions = False
        # 300 will skip all regions with area > 300
        # -300 will skipp all regions with area < 300
        self.skip_big_regions_thresh = 500

        # if > -1 then it is threshold for min intensity. Everything higher then this threshold will be skipped
        self.skip_high_intensity_regions = -1

        #TODO> WTF factor ~ INF
        #self.undefined_threshold = 0.000001
        self.undefined_threshold = 0.01
        self.weighted_matching_lost_edge_cost = 0.01

        self.mser_times = 0
        self.frame = 0

        self.fast_start = False and is_flipajs_pc()
        self.test = True

        self.auto_run = False
        self.run_to_the_end = False
        self.save_and_exit_when_finished = False


        self.intensity_threshold = 200
        self.dynamic_intensity_threshold = True
        self.dynamic_intensity_threshold_history = 5
        self.dynamic_intensity_threshold_increase = 0.0

        self.show_mser_collection = False
        self.show_ants_collection = False
        self.show_image = True
        self.show_assignment_problem = False
        self.imshow_decreasing_factor = 0.5
        self.print_mser_info = False
        self.print_matching = True

        #self.ab_area_xstart = 0.2
        #self.ab_area_ystart = 0.2
        #self.ab_area_xmax = 41
        #self.ab_area_ymax = 26
        #self.ab_area_step = 0.05
        #self.ab_area_max = 43.0

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

        self.ant_number = 5
        self.arena = my_utils.RotatedRect(my_utils.Point(986+self.border, 579+self.border), my_utils.Size(486, 486), 0)
        self.video_file_name = ''
        self.predefined_vals = ''
        self.bg = None
        self.dumpdir = os.path.expanduser('/home/flipajs/~dump/')