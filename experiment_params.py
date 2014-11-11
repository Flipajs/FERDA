__author__ = 'flip'

import cv2
import my_utils
import pickle
import os
import scipy.ndimage


class Params():
    def __init__(self):
        self.border = 0
        self.avg_ant_area = 120
        self.avg_ant_axis_a = 0
        self.avg_ant_axis_b = 0
        self.avg_ant_axis_ratio = 4.2
        self.max_axis_ratio_diff = 2
        self.max_area_diff = 0.6
        #mser min margin
        self.min_margin = 5

        self.allow_frame_seek = True

        self.inverted_image = False

        #TODO> WTF factor ~ INF
        #self.undefined_threshold = 0.000001
        self.undefined_threshold = 0.001
        self.weighted_matching_lost_edge_cost = 0.001

        self.mser_times = 0
        self.frame = 0

        self.use_gt = False
        self.gt_repair = False
        self.fast_start = False

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

        self.ant_number = 0
        self.arena = my_utils.RotatedRect(my_utils.Point(986+self.border, 579+self.border), my_utils.Size(486, 486), 0)
        self.video_file_name = ''
        self.predefined_vals = ''
        self.bg = None
        self.dumpdir = os.path.expanduser('/home/flipajs/~dump/')


        #self.dumpdir = os.path.expanduser('~/dump/Ferda')
        #self.dumpdir = os.path.expanduser('~/dump/drosophyla')
        #
        #self.ant_number = 5
        #self.arena = my_utils.RotatedRect(my_utils.Point(983+self.border, 577+self.border), my_utils.Size(476*2, 476*2), 0)
        #self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/idTracker/Messor_structor_5mayorworkers.mp4"
        #self.predefined_vals = 'messor1'
        #self.dumpdir = os.path.expanduser('~/dump/mesors')
        #self.bg = None

        #self.ant_number = 10
        #self.arena = my_utils.RotatedRect(my_utils.Point(510+self.border, 507+self.border), my_utils.Size(480*2, 480*2), 0)
        #self.video_file_name = "/home/flipajs/Downloads/movie20071009_163231_frames0001to0100_huffyuv.avi"
        #self.predefined_vals = 'octomilky'
        ##self.gt_path = '../data/eight/fixed_out.txt'
        #self.bg = None

        self.ant_number = 15
        self.arena = my_utils.RotatedRect(my_utils.Point(403+self.border, 387+self.border), my_utils.Size(767, 767), 0)
        self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/NoPlasterNoLid800/NoPlasterNoLid800.m4v"
        self.predefined_vals = 'NoPlasterNoLid800'
        self.gt_path = 'data/NoPlasterNoLid800/fixed_out.txt'
        self.bg = cv2.imread('data/noplast_bg.png')
        self.bg = scipy.ndimage.gaussian_filter(self.bg, sigma=1)
        self.dumpdir = os.path.expanduser('/home/flipajs/~dump/noplast/')
        # self.dumpdir = os.path.expanduser('~/dump/noplast')

        # self.ant_number = 8
        # self.arena = my_utils.RotatedRect(my_utils.Point(593+self.border, 570+self.border), my_utils.Size(344*2, 344*2), 0)
        # self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/eight/eight.m4v"
        # self.predefined_vals = 'eight'
        # #self.gt_path = '../data/eight/fixed_out.txt'
        # self.bg = cv2.imread('data/eight_bg.png')
        #
        # self.dumpdir = os.path.expanduser('/home/flipajs/~dump/eight')

        #self.bg2 = cv2.imread('data/eight_bg.png')

        #self.ant_number = 11
        #self.arena = my_utils.RotatedRect(my_utils.Point(665+self.border, 504+self.border), my_utils.Size(491*2, 491*2), 0)
        #self.video_file_name = "/home/flipajs/Dropbox/PycharmProjects/data/Camera 2.m4v"
        #self.predefined_vals = 'Camera2'

        #self.ant_number = 8
        #self.arena = utils.RotatedRect(utils.Point(593+self.border, 570+self.border), utils.Size(344*2, 344*2), 0)
        #self.video_file_name = ''
        #self.predefined_vals = -1