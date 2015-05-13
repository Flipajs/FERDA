__author__ = 'fnaiser'
import numpy as np


vid_path = '/Volumes/Seagate Expansion Drive/IST - videos/smallLense_colony1.avi'
working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/smalllense'
MAX_SPEED = 60
AVG_MAIN_A = 20
NODE_SIZE = 60
MIN_AREA = 50
classes = np.zeros(133)
classes[2:15] = 1
classes[44:57] = 1
classes[91:104] = 1
init_frames = 3