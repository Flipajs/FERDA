__author__ = 'flip'

import math
import ant
import numpy as np
import cv2
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
import split_by_contours
import matplotlib.pyplot as plt
import networkx as nx
import time

class ExperimentManager():
    def __init__(self, params, ants, video_manager):
        self.ant_number = params.ant_number
        self.params = params
        self.video_manager = video_manager

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
        self.img_sub_ = None
        self.prev_img_ = None
        self.dynamic_intensity_threshold = deque()
        self.groups = []
        self.groups_avg_pos = []
        self.chosen_regions_indexes = []

    def process_frame(self, img, forward=False):
        start = time.time()
        self.img_ = img.copy()
        mask = my_utils.prepare_image(img, self.params)
        self.img_sub_ = mask
        end = time.time()
        #print "time image: ", end - start

        self.params._img = img

        self.history_frame_counters(forward)
        intensity_threshold = self.get_intensity_threshold()

        if not forward:
            start = time.time()
            self.collisions = collisions.collision_detection(self.ants, self.params, self.history+1)
            end = time.time()
            #print "time collisions: ", end - start

        start = time.time()
        self.regions, self.chosen_regions_indexes = self.mser_operations.process_image(mask, intensity_threshold)
        self.groups, self.groups_avg_pos = mser_operations.get_region_groups2(self.regions)
        #self.max_margin_regions = self.get_max_margin_regions()

        self.chosen_regions_indexes = self.filter_out_children(self.chosen_regions_indexes)
        end = time.time()
        #print "timemsers etc. ", end - start

        start = time.time()
        assignment, costs, unassigned_ants, unassigned_regions = self.solve_splitting(self.chosen_regions_indexes)
        end = time.time()
        #print "time splitting: ", end - start

        #self.solve_collisions(indexes)
        #print "INDEXES: ", self.chosen_regions_indexes
        if forward and self.history < 0:
            start = time.time()
            assignment, costs = score.max_weight_matching(self.ants, self.regions, unassigned_ants, unassigned_regions, assignment, costs, self.params)
            end = time.time()
            #print "time graph1. ", end - start
            start = time.time()
            result, costs = self.solve_lost(self.ants, self.regions, self.chosen_regions_indexes, assignment, costs)
            end = time.time()
            #print "time graph2. ", end - start


        if self.params.show_assignment_problem:
            img_assignment_problem = visualize.draw_assignment_problem(self.video_manager.get_prev_img(), self.img_, self.ants, self.regions, self.chosen_regions_indexes, self.params)
            cv2.imshow("assignment problem", img_assignment_problem)
            cv2.waitKey(1)

        if forward and self.history < 0:
            self.update_ants_and_intensity_threshold(result, costs)

        self.collisions = collisions.collision_detection(self.ants, self.params, self.history)

        self.print_and_display_results()

        if forward and self.history < 0:
            self.history = 0

        #if self.params.frame == 100:
        #    print "MSER TIMES: ", self.params.mser_times

    def filter_out_children(self, indexes):
        ids = []
        for r_id in indexes:
            is_child = False
            for parent_id in indexes:
                if r_id == parent_id:
                    continue

                if mser_operations.is_child_of(self.regions[r_id], self.regions[parent_id]):
                    is_child = True
                    continue
            if not is_child:
                ids.append(r_id)

        return ids

    def prepare_graph(self, region_ids):
        graph = nx.Graph()
        thresh = 0.001

        for a in self.ants:
            graph.add_node('a'+str(a.id))
            graph.add_node('u'+str(a.id))
            graph.add_edge('a'+str(a.id), 'u'+str(a.id), weight=thresh)

        for r_id in region_ids:
            r = self.regions[r_id]

            a_area = r['area'] / float(self.params.avg_ant_area)

            for i in range(1, int(math.ceil(a_area))+1):
                graph.add_node('r'+str(r_id)+'-'+str(i))

                for a in self.ants:
                    if a.state.collision_predicted:
                        pos_p = score.position_prob_without_prediction(a, r, self.params)
                    else:
                        pos_p = score.position_prob(a, r, self.params)

                    area_p = 1
                    if i > a_area:
                        area_p = 1 + a_area - math.ceil(a_area)

                    if i == 1:
                        theta_p = score.theta_change_prob(a, r)
                        antlike_p = score.a_area_prob(r, self.params)

                        val = pos_p * area_p + theta_p * antlike_p * pos_p

                    else:
                        val = pos_p * area_p

                    if val > thresh:
                        graph.add_edge('a'+str(a.id), 'r'+str(r_id)+'-'+str(i), weight=val)

        return graph

    def get_max_margin_regions(self):
        ids = []
        for g in self.groups:
            _, id = my_utils.best_margin(self.regions, g)
            ids.append(id)

        return ids

    def solve_graph(self, graph):
        result = nx.max_weight_matching(graph, True)

        region_ids = {}
        costs = {}
        for a in self.ants:
            node = result['a'+str(a.id)]
            if node[0] == 'u':
                region_ids[a.id] = -1
                costs[a.id] = -1
            else:
                r_id, r_number = node[1:].split('-')
                region_ids[a.id] = int(r_id)
                costs[a.id] = graph.get_edge_data('a'+str(a.id), node)['weight']

        return region_ids, costs

    def split(self, region_id, ant_ids, indexes):
        points = mser_operations.prepare_region_for_splitting(self.regions[region_id], self.img_, 0.1)
        split_results = split_by_contours.solve(self.regions[region_id], points, ant_ids, self.ants, self.params, self.img_.shape, debug=True)

        return split_results

    def solve_splitting(self, indexes):
        graph = self.prepare_graph(indexes)
        ant_region_assignment, costs = self.solve_graph(graph)

        #print "ASSIGNMENT: ", ant_region_assignment
        #print "COSTS: ", costs
        assignment = [-1] * len(self.ants)

        for a in self.ants:
            a.state.in_collision = False
            a.state.in_collision_with = []


        unassigned_ants = []
        unassigned_regions = []
        for r_id in indexes:
            ant_ids = []
            for a in self.ants:
                if ant_region_assignment[a.id] == r_id:
                    ant_ids.append(a.id)

            if len(ant_ids) > 1:
                #print "SPLITTING: ", r_id, ant_ids
                split_results = self.split(r_id, ant_ids, indexes)
                for a_id in ant_ids:
                    self.ants[a_id].state.in_collision = True
                    self.ants[a_id].state.in_collision_with = ant_ids
                    unassigned_ants.append(a_id)

                self.add_new_contours(self.regions, unassigned_regions, split_results)
                self.regions[r_id]['used_for_splitting'] = True

            elif len(ant_ids) == 1:
                assignment[ant_ids[0]] = r_id
            else:
                unassigned_regions.append(r_id)

        for a in self.ants:
            if assignment[a.id] == -1:
                unassigned_ants.append(a.id)

        return assignment, costs, unassigned_ants, unassigned_regions

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

        #print "### SOLVE_LOST: l_result: ", l_result

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

        #print "#SPLITTED: ", self.number_of_splits

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

        #print " "
        #print "FRAME: ", self.params.frame

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

        #print "collisions: ", self.collisions
        #print "cg_ants_idx: ", cg_ants_idx
        #print "cg_region_groups_idx: ", cg_region_groups_idx

        #print "avg_area ", self.params.avg_ant_area

        for key in cg_ants_idx:
            result = self.solve_cg(cg_ants_idx[key], cg_region_groups_idx[key], self.groups_avg_pos)
            #if result:
            #    print "seolve_cg result: ", result
            #else:
            #    print "solve_cg result: NONE"

            #result = self.solve_to_be_split(cg_ants_idx[key], cg_region_groups_idx[key])
            for r in result:
                region_id = r[0]
                ant_ids = r[1]

                if self.regions[region_id]['area'] < (self.params.avg_ant_area / 2.):
                    continue

                if len(ant_ids) > 1:
                    self.number_of_splits += 1
                    #print "SPLITTING mser_id: ", region_id

                    points = mser_operations.prepare_region_for_splitting(self.regions[region_id], self.img_, 0.1)
                    split_results = split_by_contours.solve(self.regions[region_id], points, r[1], self.ants, self.params.frame, self.img_.shape, debug=True)
                    self.add_new_contours(self.regions, indexes, split_results)
                    self.regions[region_id]['used_for_splitting'] = True

                    #data = mser_operations.prepare_region_for_splitting(self.regions[region_id], self.img_, 0.1)
                    #new_regions = solve_merged.solve_merged(data, self.ants, r[1], self.regions[region_id]['maxI'])
                    #
                    #self.add_new_regions(self.regions, indexes, new_regions)


            #for r in result:
            #    region_id = r[0]
            #    if len(result) > 0:
            #        self.number_of_splits += 1
            #        print "SPLITTING mser_id: ", region_id
            #
            #        points = mser_operations.prepare_region_for_splitting(self.regions[region_id], self.img_, 0.1)
            #        split_results = split_by_contours.solve(self.regions[region_id], points, r[1], self.ants, self.params.frame, debug=True)
            #        self.add_new_contours(self.regions, indexes, split_results)

                    #data = self.prepare_region_for_splitting(self.regions[region_id], self.img_, 0.1)
                    #new_regions = solve_merged.solve_merged(data, self.ants, r[1], self.regions[region_id]['maxI'])
                    #
                    #self.add_new_regions(self.regions, indexes, new_regions)

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

    def add_new_contours(self, regions, indexes, new_contours):
        num = len(new_contours)
        i = 1
        for r in new_contours:
            r['label'] = regions[len(regions)-2]['label'] + 1
            r['contour'] = True

            r['margin'] = 0
            r['cx'] = r['x']
            r['cy'] = r['y']

            r['flags'] = ''

            self.groups.append([len(regions)])
            self.groups_avg_pos.append([r['cx'], r['cy']])
            indexes.append(len(regions))
            regions.append(r)

            #print "ADDED"

            i += 1

        return regions

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

    def is_antlike_region2(self, region):
        val = score.a_area_prob(region, self.params)
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

    def count_antlike_regions2(self, groups_idx):
        antlike_num = 0
        for g_idx in groups_idx:
            for r_id in self.groups[g_idx]:
                r = self.regions[r_id]
                if self.is_antlike_region2(r):
                    antlike_num += 1
                    break


        return antlike_num

    def collision_assign_ants_to_regions(self, a_ids, r_ids, r_ant_size):
        pairs = []
        for r_i in range(len(r_ids)):
            r_id = r_ids[r_i]
            r = self.regions[r_id]
            vals = [0] * len(a_ids)
            for i in range(len(a_ids)):
                a = self.ants[a_ids[i]]
                vals[i] = my_utils.e_distance(a.state.position, my_utils.Point(r['cx'], r['cy']))

            ids = np.argsort(np.array(vals))

            best_ants = []
            m = min(int(math.floor(r_ant_size[r_i])), len(a_ids))
            for i in range(m):
                best_ants.append(a_ids[ids[i]])

            pairs.append([r_id, best_ants])

        return pairs

    def solve_to_be_split(self, ants_idx, groups_idx):
        num_antlike = self.count_antlike_regions2(groups_idx)
        if num_antlike >= len(ants_idx):
            #print "Nothing to solve"
            return []

        ants_num = len(ants_idx)
        regions_ant_size_ratio = []
        regions_idx = []

        split_pairs = []
        for g_id in groups_idx:
            margin, region_id = my_utils.best_margin(self.regions, self.groups[g_id])
            regions_idx.append(region_id)
            r = self.regions[region_id]
            a = r['area'] / float(self.params.avg_ant_area)
            regions_ant_size_ratio.append(a)

        region_ant_size = 0
        for a in regions_ant_size_ratio:
            if a < 1 and a > 0.3:
                region_ant_size += 1
            else:
                region_ant_size += math.floor(a)

        #print "ants: ", ants_idx, "regions: ", regions_idx, "ras: ", region_ant_size

        increase = 0
        while ants_num > region_ant_size + increase:
            best_id = -1
            best_val = 1000
            for i in range(len(groups_idx)):
                val = math.ceil(regions_ant_size_ratio[i]) - regions_ant_size_ratio[i]
                if val < best_val:
                    best_val = val
                    best_id = i

            increase += 1

            regions_ant_size_ratio[best_id] = math.floor(regions_ant_size_ratio[best_id] + 1)

        #decrease = 0
        #while ants_num < region_ant_size - decrease:
        #    best_id = -1
        #    best_val = 1000
        #    for i in range(len(groups_idx)):
        #        val = regions_ant_size_ratio[i] - math.floor(regions_ant_size_ratio[i])
        #        if val < best_val:
        #            best_id = i
        #            best_val = val
        #
        #    decrease += 1
        #    regions_ant_size_ratio[best_id] = math.ceil(regions_ant_size_ratio[best_id] - 1)

        #if increase > 1 or decrease > 1:
        #    print "SOMETHING STRANGE HAPPEND in experiment_manager.solve_to_be_split"
        #    print "regions: ", regions_idx, " ants: ", ants_idx
        #    return []

        return self.collision_assign_ants_to_regions(ants_idx, regions_idx, regions_ant_size_ratio)

    def solve_cg(self, ants_idx, groups_idx, groups_avg_pos):
        num_antlike = self.count_antlike_regions(groups_idx)

        #if len(ants_idx) <= num_antlike:
        if num_antlike == len(groups_idx):
            #print "nothing to solve... #A: ", len(ants_idx), " num_antlike: ", num_antlike
            return []

        to_be_splitted = []

        for g_id in groups_idx:
            margin, region_id = my_utils.best_margin(self.regions, self.groups[g_id])

            if margin > 10:
                r = self.regions[region_id]
                approx_num = r['area'] / float(self.params.avg_ant_area)
                if approx_num > len(ants_idx):
                    approx_num = len(ants_idx)

                if approx_num > 1.2:
                    vals = []
                    r_p = my_utils.Point(r['cx'], r['cy'])
                    for a_id in ants_idx:
                        dist = my_utils.e_distance(self.ants[a_id].predicted_position(1), r_p)
                        vals.append([a_id, dist])

                    vals.sort(key = lambda x:x[1])
                    ids = []
                    for i in range(int(approx_num)):
                        ids.append(vals[i][0])

                    to_be_splitted.append([region_id, ids])

        return to_be_splitted

    def is_near_collision(self, cx, cy, collision):
        #TODO: near collision threshold
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
        #else:
            #print "zero ant assigned... in Experiment_manager.py"

        #print "AVG ANT AREA> ", self.params.avg_ant_area
        #print "AVG ANT AXIS RATIO> ", self.params.avg_ant_axis_ratio

    def ants_history_data(self):
        data = [None] * self.ant_number
        for i in range(self.ant_number):
            a = self.ants[i]
            vals = a.buffer_history()
            vals['moviename'] = self.params.video_file_name
            data[i] = vals

        return data

    def results_xy_vector(self):
        data = {}

        for i in range(0, self.params.frame):
            vals = {}
            for a_id in range(self.params.ant_number):
                a_frame = {}
                state = self.ants[a_id].history[self.params.frame - i - 1]

                a_frame['cx'] = state.position.x
                a_frame['cy'] = state.position.y
                a_frame['hx'] = state.head.x
                a_frame['hy'] = state.head.y
                a_frame['bx'] = state.back.x
                a_frame['by'] = state.back.y
                a_frame['certainty'] = state.score

                a_frame['lost'] = state.lost
                a_frame['in_collision'] = state.in_collision
                a_frame['in_collision_with'] = state.in_collision_with

                vals[a_id] = a_frame

            data[i] = vals

        vals = {}

        for a_id in range(self.params.ant_number):
            a_frame = {}
            state = self.ants[a_id].state

            a_frame['cx'] = state.position.x
            a_frame['cy'] = state.position.y
            a_frame['hx'] = state.head.x
            a_frame['hy'] = state.head.y
            a_frame['bx'] = state.back.x
            a_frame['by'] = state.back.y
            a_frame['certainty'] = state.score

            a_frame['lost'] = state.lost
            a_frame['in_collision'] = state.in_collision
            a_frame['in_collision_with'] = state.in_collision_with

            vals[a_id] = a_frame

        data[i+1] = vals

        return data

    def adjust_dynamic_intensity_threshold(self, max_i):
        self.dynamic_intensity_threshold.appendleft(copy.copy(self.params.intensity_threshold))
        weight = 1.0/self.params.dynamic_intensity_threshold_history
        new_val = self.params.intensity_threshold * (1-weight)
        new_val += weight * max_i
        self.params.intensity_threshold = new_val

    def add_notches(self, img):
        notch_length = 3
        steps = 250
        step = (2 * math.pi) / float(steps)
        angle = 0
        ox = self.params.arena.center.x
        oy = self.params.arena.center.y

        px = ox + self.params.arena.size.width / 2
        py = oy

        px2 = px - notch_length

        for i in range(steps):
            x = round(math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy) + ox)
            y = round(math.sin(angle) * (px - ox) - math.cos(angle) * (py - oy) + oy)

            x2 = round(math.cos(angle) * (px2 - ox) - math.sin(angle) * (py - oy) + ox)
            y2 = round(math.sin(angle) * (px2 - ox) - math.cos(angle) * (py - oy) + oy)

            cv2.line(img, (int(x), int(y)), (int(x2), int(y2)), (255, 255, 255), 1)

            angle += step
        #
        #cv2.imshow("TEST", img)
        #cv2.waitKey(0)

    #def mask_img(self, img):
    #    if self.params.bg is not None:
    #        img = self.bg_subtraction(img)
    #
    #    img = np.invert(img)
    #    img = my_utils.mask_out_arena(img, self.params.arena)
    #
    #    #self.add_notches(mask)
    #
    #    cv2.imshow("mask", img)
    #    self.img_sub_ = img
    #    return img

    def display_results(self, regions, collissions, history=0):
        img_copy = self.img_.copy()

        img_copy = visualize.draw_collision_risks(img_copy, self.ants, collissions, history)
        img_vis = visualize.draw_ants(img_copy, self.ants, regions, False, history)
        #draw_dangerous_areas(I)
        if history > 0:
            cv2.rectangle(img_vis, (0, 0), (50, 50), (255, 0, 255), -1)

        my_utils.imshow("ant track result", img_vis, self.params.imshow_decreasing_factor)
        if self.params.frame == 1:
            cv2.moveWindow("ant track result", 400, 0)

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