__author__ = 'flip'

import utils

class GroundTruth:
    def __init__(self, gt_file, experiment):
        self.f = open(gt_file)
        self.f.readline()
        self.frame = 0
        self.gt_map = []
        self.gt_precission = 10
        self.lost_counter = 0
        self.error_counter = 0
        self.right_counter = 0
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
            for a_idx in range(len(ants)):
                if utils.e_distance(ants[a_idx].state.position, utils.Point(g[0], g[1])) < self.gt_precission:
                    self.gt_map[i] = a_idx
                    break

    def check_gt(self, ants, repair=True):
        self.next_frame()
        r = [None] * len(ants)
        for i in range(len(ants)):
            g = self.next_ant()
            a = ants[self.gt_map[i]]
            if a.state.lost:
                r[i] = -1
                self.lost_counter += 1
                if a.state.lost_time > 15 and repair:
                    self.fix_error(a, utils.Point(g[0], g[1]))
            else:
                print utils.e_distance(a.state.position, utils.Point(g[0], g[1]))
                if utils.e_distance(a.state.position, utils.Point(g[0], g[1])) < self.gt_precission:
                    r[i] = 1
                    self.right_counter += 1
                else:
                    r[i] = 0
                    self.error_counter += 1
                    print a.state.position.x, a.state.position.y, g[0], g[1], a.state.area, a.state.axis_rate,
                    if repair:
                        self.fix_error(a, utils.Point(g[0], g[1]))

        return r

    def fix_error(self, ant, gt_position):
        ant.state.position = gt_position
        ant.history[0].position = gt_position



    def stats(self):
        all = self.error_counter + self.lost_counter + self.right_counter
        return "Error: " + `self.error_counter` + " (" + `self.error_counter / all` + ") Lost: " + `self.lost_counter` \
               +" (" + `self.lost_counter/all` + ") Right: " + `self.right_counter` + " (" + \
               `self.right_counter/all`+")"