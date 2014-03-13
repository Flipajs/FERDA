__author__ = 'flip'

import sys
sys.path.append('libs')
import cyMser
import cv2
import visualize
import my_utils as my_utils
from time import time
from numpy import *
import pickle


#mser flags>
# arena_kill
# max_area_diff_kill
# better_mser_nearby_kill
# axis_kill
class MserOperations():
    def __init__(self, params):
        self.mser = cyMser.PyMser()
        self.params = params

    def process_image(self, img, intensity_threshold=256):
        if img.shape[2] > 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img[:, :, 0]
            gray.shape

        t0 = time()

        if intensity_threshold > 256:
            intensity_threshold = 256
        print "THRESHOLD: ", intensity_threshold


        self.mser.process_image(gray, intensity_threshold)
        t1 = time()
        self.params.mser_times += (t1-t0)
        print 'msers takes %f' %(t1-t0)

        regions = self.mser.get_regions()

        arena_indexes = self.arena_filter(regions)

        area_indexes = self.area_filter(regions, arena_indexes)
        axis_indexes = self.axis_filter(regions, area_indexes)


        return regions, axis_indexes

    def arena_filter(self, regions):
        indexes = []
        for i in range(0, len(regions)):
            reg = regions[i]
            reg["flags"] = "arena_kill"

            if my_utils.is_inside_ellipse(self.params.arena, my_utils.Point(reg['cx'], reg['cy'])):
                indexes.append(i)
                reg["flags"] = None

        return indexes

    def area_filter(self, regions, indexes):
        filtered_indexes = []

        for i in range(len(indexes)):
            ri = regions[indexes[i]]
            d_area = ri["area"] / float(self.params.avg_ant_area)
            if d_area < self.params.max_area_diff:
                ri["flags"] = "max_area_diff_kill_small"
                continue
            else:
                d_area = float(ri["area"]) / float(self.params.avg_ant_area)
                if d_area > 1 + self.params.max_area_diff:
                    ri["flags"] = "max_area_diff_kill_big"
                    continue


            #d_area = abs(ri["area"] - self.params.avg_ant_area) / float(self.params.avg_ant_area)
            #if d_area > self.params.max_area_diff:
            #    ri["flags"] = "max_area_diff_kill"
            #    continue

            #add = True
            #for j in range(0, len(indexes)):
            #    rj = regions[indexes[j]]
            #    if my_utils.e_distance(my_utils.Point(ri['cx'], ri['cy']), my_utils.Point(rj['cx'], rj['cy'])) < 5:
            #        #if ri['area'] > rj['area']:
            #        if abs(ri['area'] - self.params.avg_ant_area) > abs(rj['area'] - self.params.avg_ant_area):
            #            add = False
            #            ri["flags"] = "better_mser_nearby_kill"
            #            break

            #if add:

            filtered_indexes.append(indexes[i])

        return filtered_indexes

    def axis_filter(self, regions, filtered_indexes):
        indexes = []
        for i in range(0, len(filtered_indexes)):
            ri = regions[filtered_indexes[i]]

            ratio, a, b = my_utils.mser_main_axis_ratio(ri['sxy'], ri['sxx'], ri['syy'])
            if abs(ratio - self.params.avg_ant_axis_ratio) < self.params.max_axis_ratio_diff:
                indexes.append(filtered_indexes[i])
            else:
                ri["flags"] = "axis_kill"

        return indexes

    def area_ratio_filter(self, regions, filtered_indexes):
        indexes = []
        for i in range(0, len(filtered_indexes)):
            ri = regions[filtered_indexes[i]]

            ratio = my_utils.mser_main_axis_ratio(ri['sxy'], ri['sxx'], ri['syy'])
            if abs(ratio - self.params.avg_ant_axis_ratio) < self.params.max_axis_ratio_diff:
                add = True

                for j in range(0, len(filtered_indexes)):
                    rj = regions[filtered_indexes[j]]
                    if my_utils.e_distance(my_utils.Point(ri['cx'], ri['cy']), my_utils.Point(rj['cx'], rj['cy'])) < 5:
                        if ri['area'] > rj['area']:
                            add = False
                            break

                if add:
                    indexes.append(filtered_indexes[i])
        return indexes

    #def ant_orientation(self, mser):
    #    theta = my_utils.mser_theta(mser["sxy"], mser["sxx"], mser["syy"])