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
import sys
import solve_merged
import collisions
from collections import deque
import matplotlib.pyplot as plt

class ExperimentManager():
    def __init__(self, params, ants):
        self.ant_number = params.ant_number
        self.params = params

        self.ants = ants
        self.number_of_splits = 0

        self.use_gt = params.use_gt
        self.regions = []
        self.history = 0
        self.collisions = []

        if self.use_gt:
            self.ground_truth = gt.GroundTruth(params.gt_path, self)
            self.ground_truth.match_gt(self.ants)

        self.mser_operations = mser_operations.MserOperations(params)
        self.count_ant_params()

        self.img_ = None
        self.dynamic_intensity_threshold = deque()
        self.groups = []
        self.groups_avg_pos = []

    def process_frame(self, img, forward=False):
        self.img_ = img.copy()
        mask = self.mask_img(img)

        self.history_frame_counters(forward)
        intensity_threshold = self.get_intensity_threshold()

        if not forward:
            self.collisions = collisions.collision_detection(self.ants, self.history+1)

        self.regions, indexes = self.mser_operations.process_image(mask, intensity_threshold)
        self.groups, self.groups_avg_pos = mser_operations.get_region_groups2(self.regions)

        self.solve_collisions(indexes)

        if forward and self.history < 0:
            result, costs = score.max_weight_matching(self.ants, self.regions, self.groups, self.params)
            result, costs = self.solve_lost(self.ants, self.regions, indexes, result, costs)
            self.update_ants_and_intensity_threshold(result, costs)

        self.collisions = collisions.collision_detection(self.ants, self.history)


        sum = 0
        for a in self.ants:
            sum += a.state.a

        print "AVG A: ", sum / float(len(self.ants))

        self.print_and_display_results()

        if forward and self.history < 0:
            self.history = 0

    def solve_lost(self, ants, regions, indexes, result, costs):
        lost_ants = []
        for i in range(self.params.ant_number):
            if result[i] < 0:
                lost_ants.append(i)

        if len(lost_ants) == 0:
            return result, costs

        free_regions = []
        for id in indexes:
            if id not in result:
                free_regions.append(id)

        l_result, l_costs = score.max_weight_matching_lost(ants, lost_ants, regions, free_regions, self.params)

        print "### SOLVE_LOST: l_result: ", l_result

        for id in range(len(lost_ants)):
            result[lost_ants[id]] = l_result[id]
            costs[lost_ants[id]] = l_costs[id]

        return result, costs


    def print_and_display_results(self):
        if self.params.show_image:
            self.display_results(self.regions, self.collisions, self.history)

        if self.params.print_mser_info:
            self.print_mser_info(self.regions)

        if self.use_gt and self.history < 0:
            r = self.ground_truth.check_gt(self.ants, self.params.gt_repair)

            self.ground_truth.display_stats()

        print "#SPLITTED: ", self.number_of_splits

    def update_ants_and_intensity_threshold(self, result, costs):
        max_i = 0
        for i in range(self.ant_number):
            if result[i] < 0:
                ant.set_ant_state_undefined(self.ants[i], result[i])
            else:
                if self.regions[result[i]]["maxI"] > max_i:
                    max_i = self.regions[result[i]]["maxI"]
                ant.set_ant_state(self.ants[i], result[i], self.regions[result[i]], cost=costs[i])

        if self.params.dynamic_intensity_threshold:
                self.adjust_dynamic_intensity_threshold(max_i)

    def get_intensity_threshold(self):
        if self.history > 0:
            return self.dynamic_intensity_threshold[self.history]

        return self.params.intensity_threshold

    def history_frame_counters(self, forward):
        if forward:
            self.params.frame += 1
            self.history -= 1
        else:
            self.history += 1
            print "history ", self.history
            self.params.frame -= 1

        print " "
        print "FRAME: ", self.params.frame

    def solve_collisions(self, indexes):
        cg_ants_idx = self.collision_groups_idx(self.collisions)

        cg_region_groups_idx = {}
        for key in cg_ants_idx:
            cg_region_groups_idx[key] = []

        for i in range(len(self.groups_avg_pos)):
            g_p = self.groups_avg_pos[i]

            near_collision, c = self.is_near_collision(g_p[0], g_p[1], self.collisions)
            if near_collision:
                for key in cg_ants_idx:
                    if c[0] in cg_ants_idx[key]:
                        cg_region_groups_idx[key].append(i)

        print "collisions: ", self.collisions
        print "cg_ants_idx: ", cg_ants_idx
        print "cg_region_groups_idx: ", cg_region_groups_idx

        print "avg_area ", self.params.avg_ant_area

        for key in cg_ants_idx:
            result = self.solve_cg(cg_ants_idx[key], cg_region_groups_idx[key], self.groups_avg_pos)
            if result:
                print "seolve_cg result: ", result
            else:
                print "solve_cg result: NONE"


            for r in result:
                region_id = r[0]
                if len(result) > 0:
                    self.number_of_splits += 1
                    print "SPLITTING mser_id: ", region_id
                    new_regions = solve_merged.solve_merged(self.regions[region_id], self.ants, r[1])

                    self.add_new_regions(self.regions, indexes, new_regions)


    def choose_region_from_group(self, regions, g, ants):
        num_a = len(ants)
        best = -1
        best_val = float('inf')

        a_area = 0
        for a in ants:
            a_area += self.ants[a].state.area

        for r_idx in self.groups[g]:
            r = regions[r_idx]
            score = abs(1 - r['area'] / float(a_area))
            if score < best_val:
                best_val = score
                best = r_idx

        return best

    def add_new_regions(self, regions, indexes, new_regions):
        num = len(new_regions)
        i = 1
        for r in new_regions:
            r['label'] = regions[len(regions)-2]['label'] + 1

            self.groups.append([len(regions)])
            self.groups_avg_pos.append([r['cx'], r['cy']])
            indexes.append(len(regions))
            regions.append(r)

            i += 1

        return regions


    def is_antlike_region(self, region):
        val = score.ab_area_prob(region, self.params)
        if val > 0.4 and region['margin'] > 18:
            return True
        else:
            return False

    def count_antlike_regions(self, groups_idx):
        antlike_num = 0
        for g_idx in groups_idx:
            for r_id in self.groups[g_idx]:
                r = self.regions[r_id]
                if self.is_antlike_region(r):
                    antlike_num += 1
                    break


        return antlike_num

    def solve_cg(self, ants_idx, groups_idx, groups_avg_pos):
        num_antlike = self.count_antlike_regions(groups_idx)

        #if len(ants_idx) <= num_antlike:
        if num_antlike == len(groups_idx):
            print "nothing to solve... #A: ", len(ants_idx), " num_antlike: ", num_antlike
            return []

        to_be_splitted = []

        for g_id in groups_idx:
            margin, region_id = my_utils.best_margin(self.regions, self.groups[g_id])

            if margin > 10:
                r = self.regions[region_id]
                approx_num = int(round(r['area'] / float(self.params.avg_ant_area)))
                if approx_num > len(ants_idx):
                    approx_num = len(ants_idx)

                if approx_num > 1:
                    vals = []
                    r_p = my_utils.Point(r['cx'], r['cy'])
                    for a_id in ants_idx:
                        dist = my_utils.e_distance(self.ants[a_id].predicted_position(1), r_p)
                        vals.append([a_id, dist])

                    vals.sort(key = lambda x:x[1])
                    ids = []
                    for i in range(approx_num):
                        ids.append(vals[i][0])

                    to_be_splitted.append([region_id, ids])

        return to_be_splitted

    def is_near_collision(self, cx, cy, collision):
        thresh = 50
        min_c = None
        min = float('inf')
        found = False
        for c in collision:
            #middle of collision
            dist = my_utils.e_distance(c[4], my_utils.Point(cx, cy))
            if dist < thresh:
                if min > dist:
                    min = dist
                    min_c = c
                found = True

        return found, min_c

    def collision_groups_idx(self, collisions):
        ant_groups = {}
        for c in collisions:
            if ant_groups.has_key(c[0]):
                ant_groups[c[0]].append(c[1])
            else:
                write0 = True
                write0_key = -1
                write1 = True
                for key in ant_groups:
                    if c[0] in ant_groups[key]:
                        write0 = False
                        write0_key = key
                    if c[1] in ant_groups[key]:
                        write1 = False
                        write1_key = key

                if write0 and write1:
                    ant_groups[c[0]] = [c[0], c[1]]
                elif write0:
                    ant_groups[write1_key].append(c[0])
                elif write1:
                    ant_groups[write0_key].append(c[1])

        return ant_groups

    def process_lost(self, lost):
        for i in range(len(self.ants)):
            if lost[i]:
                a = self.ants[i]
                if a.collision_predicted:
                    partners = self.find_collision_partners(i, lost)


    def find_collision_partners(self, a_idx, lost):
        partners = []
        for c in self.ants[a_idx].state.collisions:
            if lost[c[0]]:
                partners.append(c[0])

        return partners


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
            vals['moviename'] = self.params.video_file_name
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

        img_copy = visualize.draw_collision_risks(img_copy, self.ants, collissions, history)
        img_vis = visualize.draw_ants(img_copy, self.ants, regions, True, history)
        #draw_dangerous_areas(I)
        if history > 0:
            cv2.rectangle(img_vis, (0, 0), (50, 50), (255, 0, 255), -1)

        my_utils.imshow("ant track result", img_vis, self.params.imshow_decreasing_factor)
        if self.params.frame == 1:
            cv.MoveWindow("ant track result", 400, 0)

        if self.params.show_ants_collection:
            img_copy = self.img_.copy()
            collection = visualize.draw_ants_collection(img_copy, self.ants, history=history)
            my_utils.imshow("ants collection", collection)
        else:
            cv2.destroyWindow("ants collection")

        if self.params.show_mser_collection:
            img_copy = self.img_.copy()
            collection = visualize.draw_region_group_collection(img_copy, self.regions, self.groups, self.params)
            my_utils.imshow("mser collection", collection)
        else:
            cv2.destroyWindow("mser collection")

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

    def print_mser_info(self, regions):
        print "MSER INFO: "
        count = 0
        vals = np.zeros((len(regions), 2))
        for row in range(len(self.groups)):
            for col in range(len(self.groups[row])):
                if count % 10 == 0:
                    print ""
                i = self.groups[row][col]
                r = regions[i]
                area_p = score.area_prob(r['area'], self.params.avg_ant_area)
                axis_ratio, _, _ = my_utils.mser_main_axis_ratio(r["sxy"], r["sxx"], r["syy"])
                axis_p = score.axis_ratio_prob(axis_ratio, self.params.avg_ant_axis_ratio)
                ab = axis_ratio / self.params.avg_ant_axis_ratio
                a = r['area'] / float(self.params.avg_ant_area)

                vals[count, 0] = ab
                vals[count, 1] = a

                #print "ID: " + str(i) + " [" + str(int(r['cx'])) + ", " + str(int(r['cy'])) + "] " \
                #        "area: " + str(r['area']) + " area_p: " + str(area_p) + " label: " + str(r['label']) + \
                #      " r_size: " + str(r0) + " " + str(c0) + " " + str(r1) + " " + str(c1) + \
                #      " axis_ratio: " + str(axis_ratio) + " axis_p: " + str(axis_p) + " " + str(ab) + " " + str(a) + " " + str(r['margin'])

                if 'splitted' in r:
                    if r['splitted']:
                        print "ID: " + str(i) + " [" + str(int(r['cx'])) + ", " + str(int(r['cy'])) + "] " \
                        "area: " + str(r['area']) + " label: " + str(r['label']) + \
                      " axis_ratio: " + str(axis_ratio) + " "
                else:
                    print "ID: " + str(i) + " [" + str(int(r['cx'])) + ", " + str(int(r['cy'])) + "] " \
                        "area: " + str(r['area']) + " label: " + str(r['label']) + \
                      " axis_ratio: " + str(axis_ratio) + " minI: " + str(r['minI']) + " maxI: " + str(r['maxI']) + " margin: " + str(r['margin'])

                count += 1

        print "..........."
        print self.params.avg_ant_axis_ratio

        return vals