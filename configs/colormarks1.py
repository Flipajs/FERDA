__author__ = 'fnaiser'

import numpy as np

vid_path = '/Volumes/Seagate Expansion Drive/IST - videos/bigLenses_colormarks1.avi'
working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/biglenses1'
MAX_SPEED = 100
AVG_MAIN_A = 40
NODE_SIZE = 60
MIN_AREA = 100
ant_num=6
classes = np.zeros(597)
classes[0:6] = 1
classes[191:197] = 1
classes[383:389] = 1
init_frames = 3
