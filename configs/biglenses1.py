__author__ = 'fnaiser'

import numpy as np

vid_path = '/Users/fnaiser/Documents/Camera 1_biglense1.avi'
working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/camera1'
MAX_SPEED = 100
AVG_MAIN_A = 50
NODE_SIZE = 60
MIN_AREA = 500
ant_num=5
classes = np.zeros(1178)
classes[0:5] = 1
classes[384:389] = 1
classes[792:797] = 1
init_frames = 3
