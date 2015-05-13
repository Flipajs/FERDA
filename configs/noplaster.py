__author__ = 'fnaiser'
import numpy as np

vid_path = '/Users/fnaiser/Documents/chunks/NoPlasterNoLid800.m4v'
working_dir = '/Volumes/Seagate Expansion Drive/graphs2'
AVG_AREA = 150
AVG_MAIN_A = 25
MAX_SPEED = 60
NODE_SIZE = 50
MIN_AREA = 25
ant_num=15
classes = np.zeros(301)
classes[3:19] = 1
classes[8] = 0
classes[106:122] = 1
classes[110] = 0
classes[202:218] = 1
classes[207] = 0

init_frames=3
