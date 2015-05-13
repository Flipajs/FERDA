__author__ = 'fnaiser'
import numpy as np

vid_path = '/Users/fnaiser/Documents/chunks/eight.m4v'
working_dir = '/Volumes/Seagate Expansion Drive/mser_svm/eight'
NODE_SIZE = 50
MIN_AREA = 30
AVG_MAIN_A = 40.0
ant_num = 8
classes=np.zeros(77)
classes[0:8] = 1
classes[25:33] = 1
classes[51:59] = 1
init_frames=3