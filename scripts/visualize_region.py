__author__ = 'fnaiser'

import pickle
import cv2
from utils.video_manager import get_auto_video_manager
from utils.drawing.points import draw_points_crop


def visualize_identity(frame, id):
    working_dir='/Users/fnaiser/Documents/chunks'
    with open(working_dir+'/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)

    vid = get_auto_video_manager(working_dir+'/eight.m4v')

    for i in range(15):
        im = vid.seek_frame(frame+i)

        cv2.imshow('img', im)

        crop = draw_points_crop(im, chunks[id][frame+i].pts(), square=True)
        cv2.imshow('region', crop)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
