__author__ = 'flip'

import my_utils

class GroundTruth:
    def __init__(self, gt_file, experiment):
        #self.f = open(gt_file)
        try:
            self.f = open(gt_file)
        except IOError:
            print "Could not open file! Please close Excel!"

        self.f.readline()
        self.frame = 0
        self.gt_map = []
        self.gt_precission = 15
        self.lost_threshold = 80
        self.blinks = 0
        self.losts = 0
        self.swaps = 0
        self.exp = experiment

    def next_frame(self):
        self.f.readline()

    def next_ant(self):
        a = self.f.readline()
        a = a.split()

        return map(float, a[1:])

    def match_gt(self, ants):
        self.gt_map = [None]*len(ants)

        self.next_frame()
        for i in range(len(ants)):
            g = self.next_ant()

            best_dist = float('inf')
            best_idx = -1

            for a_idx in range(len(ants)):
                dist = my_utils.e_distance(ants[a_idx].state.position, my_utils.Point(g[0], g[1]))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = a_idx

            self.gt_map[i] = best_idx

    def check_gt(self, ants, repair=True):
        self.next_frame()
        r = [1] * len(ants)

        gt = []
        for i in range(len(ants)):
            g = self.next_ant()
            gt.append(g)

        for i in range(len(ants)):
            g = gt[i]
            a = ants[self.gt_map[i]]
            if not a.state.lost:
                if not a.state.collision_predicted:
                    swapped, swap_a_id, swap_g_id = self.is_swapped(a, gt)
                    if swapped:
                        self.swaps += 1
                        if repair:
                            self.fix_error(a, [g[0], g[1]], g[2])
                            swap_g = gt[swap_g_id]
                            self.fix_error(ants[swap_a_id], (swap_g[0], swap_g[1]), swap_g[2])

                        r[i] = -3
                        r[swap_a_id] = -3
                        continue

                elif self.is_lost(a.state.position, [g[0], g[1]]):
                        self.losts += 1
                        self.fix_error(a, [g[0], g[1]], g[2])
                        r[i] = -2
            else:
                self.blinks += 1
                r[i] = -1

    def is_swapped(self, ant, gt):
        nearest_dist = float('inf')
        nearest_id = -1

        for i in range(len(gt)):
            g = gt[i]
            g_p = my_utils.Point(g[0], g[1])
            dist = my_utils.e_distance(ant.state.position, g_p)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = i

        return ant.id != self.gt_map[nearest_id], self.gt_map[nearest_id], nearest_id


    def is_lost(self, a_p, g_p):
        g_p = my_utils.Point(g_p[0], g_p[1])

        return my_utils.e_distance(a_p, g_p) > self.lost_threshold

    #def check_gt(self, ants, repair=True):
    #    self.next_frame()
    #    r = [None] * len(ants)
    #    for i in range(len(ants)):
    #        g = self.next_ant()
    #        a = ants[self.gt_map[i]]
    #        if a.state.lost:
    #            r[i] = -1
    #            closed_interaction = False
    #            for j in range(len(ants)):
    #                if j == i:
    #                    continue
    #                test_a = ants[self.gt_map[j]].state.position
    #                if my_utils.e_distance(a.state.position, test_a) < 35:
    #                    closed_interaction = True
    #
    #            if a.state.lost_time > 50 and repair:
    #                self.fix_error(a, my_utils.Point(g[0], g[1]), g[2])
    #
    #                if not closed_interaction:
    #                    self.lost_error += 1
    #                else:
    #                    self.lost_error_in_closed_interactions += 1
    #            else:
    #                if closed_interaction:
    #                    self.lost_in_closed_interactions += 1
    #                else:
    #                    self.lost_counter += 1
    #        else:
    #            #print my_utils.e_distance(a.state.position, my_utils.Point(g[0], g[1]))
    #            if my_utils.e_distance(a.state.position, my_utils.Point(g[0], g[1])) < self.gt_precission:
    #                r[i] = 1
    #                self.right_counter += 1
    #            else:
    #                r[i] = 0
    #                self.error_counter += 1
    #                print a.state.position.x, a.state.position.y, g[0], g[1], a.state.area, a.state.axis_ratio,
    #                if repair:
    #                    self.fix_error(a, my_utils.Point(g[0], g[1]), g[2])
    #
    #    return r

    def fix_error(self, ant, gt_position, theta):
        gt_p = my_utils.Point(gt_position[0], gt_position[1])
        ant.state.position = gt_p
        ant.state.theta = theta
        ant.state.lost_time = 0
        ant.state.lost = False
        ant.state.a = self.exp.params.avg_ant_axis_a
        ant.state.b = self.exp.params.avg_ant_axis_b

        ant.history[0].position = gt_p
        ant.history[0].theta = theta

    def display_stats(self):
        print ""
        print "#BLINKS: ", self.blinks
        print "#LOSTS: ", self.losts
        print "#SWAP: ", self.swaps

    def stats(self):
        all = self.error_counter/2 + self.lost_counter + self.right_counter + self.lost_in_closed_interactions + self.lost_error_in_closed_interactions
        return "SWAP: " + `self.error_counter` + \
               "\nLost: " + `self.lost_error` + " LostInCollision: " + `self.lost_error_in_closed_interactions`\
               + "\nBlink: " + `self.lost_counter` + " Blink in Collision: " + `self.lost_in_closed_interactions` \
               + "\nRight: " + `self.right_counter` + " Total: " + `all`