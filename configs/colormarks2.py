__author__ = 'fnaiser'

import numpy as np

vid_path = '/Volumes/Seagate Expansion Drive/IST - videos/bigLenses_colormarks2.avi'
working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/biglenses2'
MAX_SPEED = 100
AVG_MAIN_A = 40
NODE_SIZE = 60
MIN_AREA = 100
ant_num=9
classes = np.zeros(848)
classes[0:9] = 1
classes[218:226] = 1
classes[430:439] = 1
classes[637:646] = 1
init_frames = 4