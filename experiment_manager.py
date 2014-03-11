__author__ = 'flip'

import math
import ant
import numpy as np
import cv2
import cv
import copy
import mser_operations
import score
import visualize
import gt
import my_utils as my_utils
import pickle
from collections import deque


# kdyz hrozi kolize, mnohem ostrejsi pravidla na fit...
# jinak muze byt clovek celkem benevolentni...


class ExperimentManager():
    def __init__(self, params, ants):
        self.ant_number = params.ant_number
        self.params = params

        self.ants = ants

        self.use_gt = False
        self.regions = []
        self.history = 0

        if self.use_gt:
            self.ground_truth = gt.GroundTruth('fixed_out.txt', self)
            self.ground_truth.match_gt(self.ants)

        self.mser_operations = mser_operations.MserOperations(params)
        self.count_ant_params()

        self.img_ = None
        self.dynamic_intensity_threshold = deque()

    def process_frame(self, img, forward=False):
        self.img_ = img.copy()
        mask = self.mask_img(img)


        print "pre: ", self.history
        if forward:
            self.params.frame += 1
            self.history -= 1
        else:
            self.history += 1
            print "history ", self.history
            self.params.frame -= 1

        intensity_threshold = self.params.intensity_threshold
        if self.history > 0:
            intensity_threshold = self.dynamic_intensity_threshold[self.history]

        self.regions, indexes = self.mser_operations.process_image(mask, intensity_threshold)
        if forward and self.history < 0:
            print "fwd + detect"
            self.history = 0
            result = score.max_weight_matching(self.ants, self.regions, indexes, self.params)

            max_i = 0
            for i in range(self.ant_number):
                if result[i] < 0:
                    ant.set_ant_state_undefined(self.ants[i], result[i])
                else:
                    if self.regions[result[i]]["maxI"] > max_i:
                        max_i = self.regions[result[i]]["maxI"]
                    ant.set_ant_state(self.ants[i], result[i], self.regions[result[i]])

            if self.params.dynamic_intensity_threshold:
                self.adjust_dynamic_intensity_threshold(max_i)

            collissions = self.collission_detection()

        self.display_results(self.regions, collissions, self.history)

    def collission_detection(self):
        thresh = 70

        collisions = []
        for i in range(len(self.ants)):
            a1 = self.ants[i]
            for j in range(i+1, len(self.ants)):
                a2 = self.ants[j]

                a1cx = a1.state.position.x
                a1cy = a1.state.position.y
                a2cx = a2.state.position.x
                a2cy = a2.state.position.y

                x = a1cx - a2cx
                y = a1cy - a2cy

                dist = math.sqrt(x*x + y*y)
                if dist < thresh:
                    #print a1.state.back.x, a1.state.back.y
                    #print a1.state.head.x, a1.state.head.y

                    collisions.append((i, j, dist))

        return collisions

    def count_ant_params(self):
        avg_area = 0
        avg_axis_ratio = 0
        avg_axis_a = 0
        avg_axis_b = 0
        counter = 0
        for a in self.ants:
            if a.state.mser_id == -1:
                continue

            avg_area += a.state.area
            avg_axis_ratio += a.state.axis_ratio
            avg_axis_a += a.state.a
            avg_axis_b += a.state.b
            counter += 1

        if counter > 0:
            self.params.avg_ant_area = avg_area / counter
            self.params.avg_ant_axis_ratio = avg_axis_ratio / counter
            self.params.avg_ant_axis_a = avg_axis_a / counter
            self.params.avg_ant_axis_b = avg_axis_b / counter
        else:
            print "zero ant assigned... in Experiment_manager.py"

        print "AVG ANT AREA> ", self.params.avg_ant_area
        print "AVG ANT AXIS RATIO> ", self.params.avg_ant_axis_ratio

    def ants_history_data(self):
        data = [None] * self.ant_number
        for i in range(self.ant_number):
            a = self.ants[i]
            vals = a.buffer_history()
            vals['moveiname'] = self.params.video_file_name
            data[i] = vals

        return data

    def adjust_dynamic_intensity_threshold(self, max_i):
        self.dynamic_intensity_threshold.appendleft(copy.copy(self.params.intensity_threshold))
        weight = 1.0/self.params.dynamic_intensity_threshold_history
        new_val = self.params.intensity_threshold * (1-weight)
        new_val += weight * max_i
        self.params.intensity_threshold = new_val

    def mask_img(self, img):
        mask = np.ones((np.shape(img)[0], np.shape(img)[1], 1), dtype=np.uint8)*255
        cv2.circle(mask, self.params.arena.center.int_tuple(), self.params.arena.size.width/2, 0, -1)
        idx = (mask == 0)
        mask[idx] = self.img_[idx]

        return mask

    def display_results(self, regions, collissions, history=0):
        img_copy = self.img_.copy()

        img_copy = visualize.draw_collission_risks(img_copy, self.ants, collissions)
        img_vis = visualize.draw_ants(img_copy, self.ants, regions, True, history)
        #draw_dangerous_areas(I)
        my_utils.imshow("ant track result", img_vis, self.params.imshow_decreasing_factor)
        if self.params.frame == 1:
            cv.MoveWindow("ant track result", 400, 0)
        if self.params.show_mser_collection:
            img_copy = self.img_.copy()
            collection = visualize.draw_region_collection(img_copy, regions, self.params)
            my_utils.imshow("mser collection", collection)
        else:
            cv2.destroyWindow("mser collection")

        if self.params.show_ants_collection:
            img_copy = self.img_.copy()
            collection = visualize.draw_ants_collection(img_copy, self.ants, history=history)
            my_utils.imshow("ants collection", collection)
        else:
            cv2.destroyWindow("ants collection")

        if self.use_gt:
            r = self.ground_truth.check_gt(self.ants, True)
            if r.count(0) > 0:
                broken_idx = [i for i in range(len(r)) if r[i] == 0]
                for i in broken_idx:
                    print self.ants[i].state

            print self.ground_truth.stats()

    def save_ants_info(self, regions):
        img_copy = self.img_.copy()

        collection = visualize.draw_region_collection(img_copy, regions, self.params)
        cv2.imwrite("out/collection_"+str(self.params.frame)+".png", collection)
        afile = open(r'out/regions_'+str(self.params.frame)+'pkl', 'wb')
        pickle.dump(regions, afile)
        afile.close()

        ants = [None]*self.params.ant_number
        for i in range(self.params.ant_number):
            ants[i] = self.ants[i].state

        afile = open(r'out/ants_'+str(self.params.frame)+'.pkl', 'wb')
        pickle.dump(ants, afile)
        afile.close()

    def log_regions_collection(self):
        img_copy = self.img_.copy()

        collection = visualize.draw_region_collection(img_copy, self.regions, self.params)
        cv2.imwrite("out/collisions/collection_"+str(self.params.frame)+".png", collection)

    def log_regions(self):
        afile = open(r'out/collisions/regions_'+str(self.params.frame)+'pkl', 'wb')
        pickle.dump(self.regions, afile)
        afile.close()

    def log_frame(self):
        cv2.imwrite("out/frames/frame"+str(self.params.frame)+".png", self.img_)