# coding=utf8
import pickle
from sys import maxint
import math
import os
from PyQt4 import QtCore

import numpy

import default_settings


class IdentityManager():
    """A class that deals with ant's positions, identities, history and such things."""

    def __init__(self, filepath):
        self.switch_causes = {'proximity', 'collision', 'lost', 'overlap'}

        self.safety_max = 2
        self.max_certainty = 0
        self.faults = dict()
        self.switch_faults = dict()
        self.non_switch_faults = dict()
        self.changes = []
        self.changes_for_frames = dict()
        self.loaded_change_files = set()
        self.change_positions = dict()
        self.change_index = 0
        self.frame_numbers = []
        self.results = None
        self.ant_num = 0
        self.group_num = 2
        self.fault_num = 0
        self.info = None
        self.previously_loaded_changes = []

        self.max_frame = -1;

        self.load_results(filepath)

        self.find_max_certainty()
        self.normalize_certainties(self.max_certainty)

        settings = QtCore.QSettings("Ants correction tool")

        if settings.value('head_detection', default_settings.get_default('head_detection'), bool):
            self.pair_points()
            self.detect_heads()

        if settings.value('undo_redo_mode', default_settings.get_default('undo_redo_mode'), str) == 'separate':
            self.write_change = self.write_change_frame_by_frame
            self.undo_change = self.undo_change_frame_by_frame
            self.redo_change = self.redo_change_frame_by_frame
        elif settings.value('undo_redo_mode', default_settings.get_default('undo_redo_mode'), str) == 'global':
            self.write_change = self.write_change_all_in_one
            self.undo_change = self.undo_change_all_in_one
            self.redo_change = self.redo_change_all_in_one

        self.undo_redo_mode = settings.value('undo_redo_mode', default_settings.get_default('undo_redo_mode'), str)

        self.set_new_settings()

    def load_results(self, filepath):
        """Loads results from the given filepath."""
        pfile = open(filepath, 'rb')
        results = pickle.load(pfile)
        pfile.close()
        self.ant_num = results['info']['ant_number']
        self.info = results.pop('info', None)
        self.results = results
        self.max_speeds = [None]*self.ant_num
        self.avg_lengths = [None]*self.ant_num

        self.max_frame = max(self.results.keys())

    def get_max_frame(self):
        return self.max_frame

    def get_positions(self, frame_num, ant_num):
        """Returns positions of given ant in given frame"""
        if frame_num in self.results and ant_num in self.results[frame_num]:
            res = self.results[frame_num][ant_num]
        else:
            res = None

        return res

    def get_group(self, ant_num):
        """Returns group of an ant. Placeholder implementation."""
        return int(round(float(ant_num)/self.ant_num))

    def save(self, filepath):
        """Saves results into given file."""
        pfile = open(filepath, 'wb')
        self.results['info'] = self.info
        pickle.dump(self.results, pfile)
        self.results.pop('info', None)
        pfile.close()

    def get_video_file_name(self):
        """Returns name of the video file the results correspond to"""
        return self.results['info']['video_file_name']

    def find_max_certainty(self):
        """Determines maximal certainty in the results. Ignores those that are bigger than self.safety_max."""
        self.max_certainty = 0
        for i in self.results:
            for j in range(self.ant_num):
                if self.max_certainty < self.results[i][j]['certainty'] <= self.safety_max:
                    self.max_certainty = self.results[i][j]['certainty']

    def get_faulty_frames(self):
        """Returns a sorted list of such frames in which faults occur."""
        return sorted(self.faults.keys())

    def get_faulty_switch_frames(self):
        """Returns a sorted list of such frames in which possible switches occur."""
        return sorted(self.switch_faults.keys())

    def get_faulty_non_switch_frames(self):
        """Returns a sorted list of such frames in which possible faults that are not switches occur."""
        return sorted(self.non_switch_faults.keys())

    def get_switch_faults(self, frame_no):
        """Returns all possible switches in frame given."""
        return self.switch_faults[frame_no]

    def get_non_switch_faults(self, frame_no):
        """Returns all possible faults that are not switches in frame given."""
        return self.non_switch_faults[frame_no]

    def get_faults(self, frame_no):
        """Returns all faults occurring in given frame"""
        return self.faults[frame_no]

    def get_fault_num(self):
        """Return total number of faults"""
        return self.fault_num

    def pair_points(self):
        """Determines which points (tail or head) of one ant belong together in successive frames. In the end, all points
        marked h should correspond to same end of an ant, and the same for points marked b. The reasoning is following:
        all possible distances between point in frame i and frame i + 1 are calculated. The final configuration is selected
        such that all those distances are close to eachother."""
        last_heads = [[0, 0] for j in range(self.ant_num)]
        last_centers = [[0, 0] for j in range(self.ant_num)]
        last_backs = [[0, 0] for j in range(self.ant_num)]
        for i in sorted(self.results):
            for j in range(self.ant_num):
                head_head_dist = self.distance(last_heads[j][0], last_heads[j][1], self.results[i][j]['hx'], self.results[i][j]['hy'])
                head_back_dist = self.distance(last_heads[j][0], last_heads[j][1], self.results[i][j]['bx'], self.results[i][j]['by'])
                back_head_dist = self.distance(last_backs[j][0], last_backs[j][1], self.results[i][j]['hx'], self.results[i][j]['hy'])
                back_back_dist = self.distance(last_backs[j][0], last_backs[j][1], self.results[i][j]['bx'], self.results[i][j]['by'])
                center_center_dist = self.distance(last_centers[j][0], last_centers[j][1], self.results[i][j]['cx'], self.results[i][j]['cy'])

                sign_one = abs(center_center_dist - head_back_dist) + abs(center_center_dist - back_head_dist)
                sign_two = abs(center_center_dist - head_head_dist) + abs(center_center_dist - back_back_dist)

                if sign_one < sign_two:
                    self.swap_tail_head(i, j)

                #Save new positions
                last_heads[j] = [self.results[i][j]['hx'], self.results[i][j]['hy']]
                last_centers[j] = [self.results[i][j]['cx'], self.results[i][j]['cy']]
                last_backs[j] = [self.results[i][j]['bx'], self.results[i][j]['by']]

    def detect_heads(self):
        """Tries to determine which of the two points marked b and h is head. (these point need to be paired) It does so
        by counting in which direction the ant goes mostly.
        """
        dir_sums = [0 for j in range(self.ant_num)]
        last_centers = [[0, 0] for j in range(self.ant_num)]
        for i in sorted(self.results):
            for j in range(self.ant_num):

                if self.distance(self.results[i][j]['hx'], self.results[i][j]['hy'], last_centers[j][0], last_centers[j][1]) > \
                    self.distance(self.results[i][j]['bx'], self.results[i][j]['by'], last_centers[j][0], last_centers[j][1]):
                    dir_sums[j] += 1
                else:
                    dir_sums[j] -= 1

                last_centers[j] = [self.results[i][j]['cx'], self.results[i][j]['cy']]

        for j in range(self.ant_num):
            if dir_sums[j] < 0:
                self.swap_tail_head_from_frame(0, j)

    def to_line(self, hx, hy, bx, by, x, y):
        return (hx-bx)*x + (hy-by)*y + hx*bx + hy*hy - hx**2 - hy**2

    def swap_tail_head(self, frame_no, ant_no):
        """Swaps tail and head of given and in given frame"""
        tmp = self.results[frame_no][ant_no]['hx']
        self.results[frame_no][ant_no]['hx'] = self.results[frame_no][ant_no]['bx']
        self.results[frame_no][ant_no]['bx'] = tmp

        tmp = self.results[frame_no][ant_no]['hy']
        self.results[frame_no][ant_no]['hy'] = self.results[frame_no][ant_no]['by']
        self.results[frame_no][ant_no]['by'] = tmp

    def swap_tail_head_from_frame(self, frame_from, ant_no):
        """Swaps tail and head in all frames starting at the frame given up till the end of data."""
        for frame_no in range(frame_from, len(self.results)):
            self.swap_tail_head(frame_no, ant_no)

    def compute_distance(self, frame_no, ant_one, ant_two, key_x_one, key_y_one, key_x_two, key_y_two):
        """Computes distance of ant points with given parameters. """
        return ((self.results[frame_no][ant_one][key_x_one] - self.results[frame_no][ant_two][key_x_two])**2 + (self.results[frame_no][ant_one][key_y_one] - self.results[frame_no][ant_two][key_y_two])**2)**.5

    def distance(self, x1, y1, x2, y2):
        return ((x1 - x2)**2 + (y1 - y2)**2)**.5

    def normalize_certainties(self, max_certainty):
        """Normalizes all certainties to fit interval <0, 1>"""
        for i in self.results:
            for j in range(self.ant_num):
                self.results[i][j]['certainty'] = self.results[i][j]['certainty']/max_certainty

    def write_change_frame_by_frame(self, frame_number, cause, change_data):
        """Writes a performed change. Used when the history mode is frame by frame"""
        if frame_number in self.results:
            old_data = None
            if cause == 'movement':
                old_data = dict()
                for ant_id in change_data:
                    old_data[ant_id] = {'cx': None, 'cy': None, 'hx': None, 'hy': None, 'bx': None, 'by': None}
                    for key in ['cx', 'cy', 'hx', 'hy', 'bx', 'by']:
                        if change_data[ant_id][key] is not None:
                            old_data[ant_id][key] = self.results[frame_number][ant_id][key]
                            self.results[frame_number][ant_id][key] = change_data[ant_id][key]
            elif cause == 'swap':
                self.swap_from_frame(frame_number, change_data[0], change_data[1])

            if frame_number not in self.changes_for_frames:
                self.changes_for_frames[frame_number] = []
                self.change_positions[frame_number] = 0

            change = {'cause': cause, 'change_data': change_data, 'old_data': old_data}

            if self.change_positions[frame_number] != len(self.changes_for_frames[frame_number]):
                for i in range(self.change_positions[frame_number], len(self.changes_for_frames[frame_number])):
                    self.frame_numbers.pop(self.changes.index(self.changes_for_frames[frame_number][i]))
                    self.changes.remove(self.changes_for_frames[frame_number][i])
                del self.changes_for_frames[frame_number][self.change_positions[frame_number]:]
            self.change_positions[frame_number] += 1
            self.changes_for_frames[frame_number].append(change)
            self.changes.append(change)
            self.frame_numbers.append(frame_number)

    def undo_change_frame_by_frame(self, frame_number):
        """Undoes last performed change in given frame."""
        if frame_number in self.changes_for_frames:
            if self.change_positions[frame_number] > 0:
                self.change_positions[frame_number] -= 1
                last_change = self.changes_for_frames[frame_number][self.change_positions[frame_number]]
                if last_change['cause'] == 'movement':
                    for ant_id in last_change['old_data']:
                        for key in ['cx', 'cy', 'hx', 'hy', 'bx', 'by']:
                            if last_change['old_data'][ant_id][key] is not None:
                                self.results[frame_number][ant_id][key] = last_change['old_data'][ant_id][key]
                elif last_change['cause'] == 'swap':
                    self.swap_from_frame(frame_number, last_change['change_data'][0], last_change['change_data'][1])

    def redo_change_frame_by_frame(self, frame_number):
        """Redoes last performed change in given frame."""
        if frame_number in self.changes_for_frames:
            if self.change_positions[frame_number] < len(self.changes_for_frames[frame_number]):
                last_change = self.changes_for_frames[frame_number][self.change_positions[frame_number]]
                self.change_positions[frame_number] += 1
                if last_change['cause'] == 'movement':
                    for ant_id in last_change['change_data']:
                        for key in ['cx', 'cy', 'hx', 'hy', 'bx', 'by']:
                            if last_change['change_data'][ant_id][key] is not None:
                                self.results[frame_number][ant_id][key] = last_change['change_data'][ant_id][key]
                elif last_change['cause'] == 'swap':
                    self.swap_from_frame(frame_number, last_change['change_data'][0], last_change['change_data'][1])

    def write_change_all_in_one(self, frame_number, cause, change_data):
        """Writes a performed change. Used when the history mode is global."""
        if frame_number in self.results:
            old_data = None
            if cause == 'movement':
                old_data = dict()
                for ant_id in change_data:
                    old_data[ant_id] = {'cx': None, 'cy': None, 'hx': None, 'hy': None, 'bx': None, 'by': None}
                    for key in ['cx', 'cy', 'hx', 'hy', 'bx', 'by']:
                        if change_data[ant_id][key] is not None:
                            old_data[ant_id][key] = self.results[frame_number][ant_id][key]
                            self.results[frame_number][ant_id][key] = change_data[ant_id][key]
            elif cause == 'swap':
                self.swap_from_frame(frame_number, change_data[0], change_data[1])

            change = {'cause': cause, 'change_data': change_data, 'old_data': old_data}

            if self.change_index != len(self.changes):
                del self.changes[self.change_index:]
                del self.frame_numbers[self.change_index:]
            self.changes.append(change)
            self.frame_numbers.append(frame_number)
            self.change_index += 1

    def undo_change_all_in_one(self, *args):
        """Undoes last performed change"""
        if self.change_index > 0:
            self.change_index -= 1
            last_change = self.changes[self.change_index]
            frame_number = self.frame_numbers[self.change_index]
            if last_change['cause'] == 'movement':
                    for ant_id in last_change['old_data']:
                        for key in ['cx', 'cy', 'hx', 'hy', 'bx', 'by']:
                            if last_change['old_data'][ant_id][key] is not None:
                                self.results[frame_number][ant_id][key] = last_change['old_data'][ant_id][key]
            elif last_change['cause'] == 'swap':
                self.swap_from_frame(frame_number, last_change['change_data'][0], last_change['change_data'][1])

    def redo_change_all_in_one(self, *args):
        """Redoes last undone change."""
        if self.change_index < len(self.changes):
            frame_number = self.frame_numbers[self.change_index]
            last_change = self.changes[self.change_index]
            if last_change['cause'] == 'movement':
                for ant_id in last_change['change_data']:
                    for key in ['cx', 'cy', 'hx', 'hy', 'bx', 'by']:
                        if last_change['change_data'][ant_id][key] is not None:
                            self.results[frame_number][ant_id][key] = last_change['change_data'][ant_id][key]
            elif last_change['cause'] == 'swap':
                self.swap_from_frame(frame_number, last_change['change_data'][0], last_change['change_data'][1])
            self.change_index += 1

    def write_change(self, frame_number, cause, change_data):
        """Method that is used for writing changes. Note that it serves just as a pointer to one of the
        write_change_<mode> functions.
        """
        raise RuntimeError

    def undo_change(self, frame_number):
        """Method that is used for writing changes. Note that it serves just as a pointer to one of the
        undo_change_<mode> functions.
        """
        raise RuntimeError

    def redo_change(self, frame_number):
        """Method that is used for writing changes. Note that it serves just as a pointer to one of the
        redo_change_<mode> functions.
        """
        raise RuntimeError

    def swap_identities(self, frame_number, ant_one, ant_two):
        """Swaps the two ants given in the frame given."""
        if frame_number in self.results:
            tmp = self.results[frame_number][ant_one]
            self.results[frame_number][ant_one] = self.results[frame_number][ant_two]
            self.results[frame_number][ant_two] = tmp

    def swap_from_frame(self, frame_number, ant_one, ant_two):
        """Swaps the two ants given in frames from frame given up to the end of data."""
        current_frame = frame_number
        while current_frame in self.results:
            self.swap_identities(current_frame, ant_one, ant_two)
            current_frame += 1

    def faults_to_txt_file(self, filename):
        """Saves numbers of faulty frames to given text file"""
        file = open(filename, "wt")
        for i in self.get_faulty_frames():
            file.write(str(i) + ' ')
        file.close()

    def changes_to_file(self, filename):
        """Saves performed changes to given file. The format is the same for both of history modes"""
        complete_changes = []
        if self.undo_redo_mode == 'separate':
            for frame_number in self.changes_for_frames:
                for i in range(self.change_positions[frame_number]):
                    complete_changes.append([frame_number, self.changes_for_frames[frame_number][i]])
        elif self.undo_redo_mode == 'global':
            for i in range(self.change_index):
                complete_changes.append([self.frame_numbers[i], self.changes[i]])
        complete_changes.extend(self.previously_loaded_changes)
        file = open(filename, 'wb')
        pickle.dump(complete_changes, file)
        file.close()

    def changes_from_file(self, filename):
        """Loads changes from given file. Applies those changes."""
        if os.path.abspath(filename) not in self.loaded_change_files:
            self.loaded_change_files.add(os.path.abspath(filename))
            file = open(filename, 'rb')
            changelist = pickle.load(file)
            file.close()
            for changelistitem in changelist:
                if changelistitem[1]['cause'] == 'movement':
                    for ant_id in changelistitem[1]['change_data']:
                        for key in ['cx', 'cy', 'hx', 'hy', 'bx', 'by']:
                            if changelistitem[1]['change_data'][ant_id][key] is not None:
                                self.results[changelistitem[0]][ant_id][key] = changelistitem[1]['change_data'][ant_id][key]
                elif changelistitem[1]['cause'] == 'swap':
                    self.swap_from_frame(changelistitem[0], changelistitem[1]['change_data'][0], changelistitem[1]['change_data'][1])
            self.previously_loaded_changes.extend(changelist)


    def set_new_settings(self):
        """This method should be called when the settings that are relevant for identity_manager were changed. Recalculates
        all that is needed to adapt to new settings"""
        settings = QtCore.QSettings("Ants correction tool")
        self.tester = FaultTester(self, self.switch_causes, len_test=settings.value('len_test', default_settings.get_default('len_test'), bool), certainty_test=settings.value('certainty_test', default_settings.get_default('certainty_test'), bool), proximity_test=settings.value('proximity_test', default_settings.get_default('proximity_test'), bool), angular_test=settings.value('angular_test', default_settings.get_default('angular_test'), bool), lost_test=settings.value('lost_test', default_settings.get_default('lost_test'), bool), collision_test=settings.value('collision_test', default_settings.get_default('collision_test'), bool), overlap_test=settings.value('overlap_test', default_settings.get_default('overlap_test'), bool))
        self.tester.set_tolerances(len_tolerance=settings.value('length_tolerance', default_settings.get_default('length_tolerance'), float), min_certainty=settings.value('minimal_certainty', default_settings.get_default('minimal_certainty'), float), proximity_tolerance=settings.value('proximity_tolerance', default_settings.get_default('proximity_tolerance'), float), angular_tolerance=settings.value('angular_tolerance', default_settings.get_default('angular_tolerance'), int))
        old_undo_redo_mode = self.undo_redo_mode
        self.undo_redo_mode = settings.value('undo_redo_mode', default_settings.get_default('undo_redo_mode'), str)
        if old_undo_redo_mode != self.undo_redo_mode:
            if old_undo_redo_mode == 'global':
                self.write_change = self.write_change_frame_by_frame
                self.undo_change = self.undo_change_frame_by_frame
                self.redo_change = self.redo_change_frame_by_frame
                self.change_positions = dict()
                self.changes_for_frames = dict()
                del self.changes[self.change_index:]
                del self.frame_numbers[self.change_index:]
                for i in range(self.change_index):
                    if self.frame_numbers[i] not in self.changes_for_frames:
                        self.changes_for_frames[self.frame_numbers[i]] = []
                        self.change_positions[self.frame_numbers[i]] = 0
                    self.changes_for_frames[self.frame_numbers[i]].append(self.changes[i])
                    self.change_positions[self.frame_numbers[i]] += 1
            elif old_undo_redo_mode == 'separate':
                self.write_change = self.write_change_all_in_one
                self.undo_change = self.undo_change_all_in_one
                self.redo_change = self.redo_change_all_in_one
                self.change_index = 0
                for frame_number in self.changes_for_frames:
                    for i in range(self.change_positions[frame_number], len(self.changes_for_frames[frame_number])):
                        self.frame_numbers.pop(self.changes.index(self.changes_for_frames[frame_number][i]))
                        self.changes.remove(self.changes_for_frames[frame_number][i])
                    self.change_index += self.change_positions[frame_number]

    def compute_faults(self):
        """Computes possible faults"""
        self.faults, self.switch_faults, self.non_switch_faults = self.tester.compute_faults()
        self.fault_num = self.tester.fault_num
        # self.faults_to_txt_file(u'C:\\Users\\Míma\\Documents\\CMP\\fault_detects.txt')


class FaultTester(object):
    """This class is used to look for faulty frames. All tests can be switched on and off. The testing system works
    as follows: first, all methods in self.inits are performed. Than the tester iterates through all ants and runs
    methods in self.init_runs. That this run needs to be processed, so all methods in self.process_init_runs are called.
    Lastly, another run is performed. Once again it is iterated through all the ants calling all method in self.decision_tests.
    Should you add another test, add a boolean into constructor. If this boolean is true, add methods of your test to
    some or all method lists already described. The test_decide method should call self.identity_manager.add_fault if it
    finds a fault.
    Docstrings of tests are written into decision methods.
    Itits and process inits methods have no parameters, init_runs and decision_test have parametres (frame, ant)
    """

    def __init__(self, identity_manager, switch_causes, len_test=True, certainty_test=True, proximity_test=True, angular_test=True, lost_test=True, collision_test=True, overlap_test=True):
        self.switch_causes = switch_causes

        self.len_tolerance = 2
        self.min_certainty = .2
        self.proximity_tolerance = 5
        self.angular_tolerance = 45

        self.identity_manager = identity_manager
        self.faults = dict()
        self.switch_faults = dict()
        self.non_switch_faults = dict()
        self.inits = []
        self.init_runs = []
        self.process_init_runs = []
        self.decision_tests = []
        self.frame_count = 0
        self.fault_num = 0

        if len_test:
            self.inits.append(self.len_test_init)
            self.init_runs.append(self.len_test_init_run)
            self.process_init_runs.append(self.len_test_process_init_run)
            self.decision_tests.append(self.len_test_decision)

        if certainty_test:
            self.inits.append(self.certainty_test_init)
            self.decision_tests.append(self.certainty_test_decision)

        if proximity_test:
            self.decision_tests.append(self.proximity_test_decision)

        if angular_test:
            self.inits.append(self.angular_test_init)
            self.decision_tests.append(self.angular_test_decision)

        if lost_test:
            self.decision_tests.append(self.lost_test_decision)

        if collision_test:
            self.decision_tests.append(self.collision_test_decision)

        if overlap_test:
            self.decision_tests.append(self.overlap_test_decision)

    #Inits
    def len_test_init(self):
        self.average_len = 0
        self.len_sums = [0 for i in range(self.identity_manager.ant_num)]
        self.avg_lengths = [0 for i in range(self.identity_manager.ant_num)]

    def certainty_test_init(self):
        self.len_sums = [0 for i in range(self.identity_manager.ant_num)]

    def angular_test_init(self):
        self.last_heads = [[self.identity_manager.results[0][j]['hx'], self.identity_manager.results[0][j]['hy']] for j in range(self.identity_manager.ant_num)]
        self.last_backs = [[self.identity_manager.results[0][j]['bx'], self.identity_manager.results[0][j]['by']] for j in range(self.identity_manager.ant_num)]

    #Init runs
    def len_test_init_run(self, i, j):
        self.len_sums[j] += ((self.identity_manager.results[i][j]['hx'] - self.identity_manager.results[i][j]['bx'])**2 + (self.identity_manager.results[i][j]['hy'] - self.identity_manager.results[i][j]['by'])**2)**.5

    #Process init runs
    def len_test_process_init_run(self):
        for j in range(self.identity_manager.ant_num):
            self.avg_lengths[j] = self.len_sums[j]/self.frame_count

    #Decicion runs
    def len_test_decision(self, i, j):
        """Checks whether the ant's length is self.len_tolerance times bigger or smaller than the average length of that ant"""
        ant_len = ((self.identity_manager.results[i][j]['hx'] - self.identity_manager.results[i][j]['bx'])**2 + (self.identity_manager.results[i][j]['hy'] - self.identity_manager.results[i][j]['by'])**2)**.5
        if ant_len < self.avg_lengths[j]*(1./self.len_tolerance) or ant_len > self.avg_lengths[j]*self.len_tolerance:
            self.add_fault(i, 'len', j)

    def certainty_test_decision(self, i, j):
        """Checks whether the certainty of an ant is bigger than self.min_certainty."""
        if self.identity_manager.results[i][j]['certainty'] < self.min_certainty:
            self.add_fault(i, 'certainty', j)

    def proximity_test_decision(self, i, j):
        """Checks whether there is any and that is too close to this one"""
        for k in range(j + 1, self.identity_manager.ant_num):
            if self.combined_proximity(i, j, k) < self.proximity_tolerance:
                self.add_fault(i, 'proximity', {j, k})

    def angular_test_decision(self, i, j):
        """Doesn't seem to detect any fault at all, although it works as it should. It checks whether the ant doesn't turn
        too fast
        """
        if min(math.degrees(self.angle([self.last_heads[j][0] - self.last_backs[j][0], self.last_heads[j][1] - self.last_backs[j][1]], [self.identity_manager.results[i][j]['hx'] - self.identity_manager.results[i][j]['bx'], self.identity_manager.results[i][j]['hy'] - self.identity_manager.results[i][j]['by']])),
                math.degrees(self.angle([self.last_heads[j][0] - self.last_backs[j][0], self.last_heads[j][1] - self.last_backs[j][1]], [self.identity_manager.results[i][j]['bx'] - self.identity_manager.results[i][j]['hx'], self.identity_manager.results[i][j]['by'] - self.identity_manager.results[i][j]['hy']]))) \
                > self.angular_tolerance:
            self.add_fault(i, 'angle', j)

        self.last_heads[j][0] = self.identity_manager.results[i][j]['hx']
        self.last_heads[j][1] = self.identity_manager.results[i][j]['hy']
        self.last_backs[j][0] = self.identity_manager.results[i][j]['bx']
        self.last_backs[j][1] = self.identity_manager.results[i][j]['by']

    def lost_test_decision(self, i, j):
        """Checks whether the tracker didn't loose the ant"""
        try:
            if self.identity_manager.results[i][j]['lost']:

                self.add_fault(i, 'lost', j)

        except KeyError:
            pass

    def collision_test_decision(self, i, j):
        """Checks for the in_collision_with parameter from tracker"""
        try:
            if self.identity_manager.results[i][j]['in_collision_with']:

                involved_ants = set(self.identity_manager.results[i][j]['in_collision_with'])
                involved_ants.add(j)
                self.add_fault(i, 'collision', involved_ants)
        except KeyError:
            pass

    def overlap_test_decision(self, i, j):
        """Draws a square around each ant. Checks whether some of those squares overlap"""
        ant_one = QtCore.QRectF(QtCore.QPointF(min(self.identity_manager.results[i][j]['cx'], self.identity_manager.results[i][j]['hx'], self.identity_manager.results[i][j]['bx']),
                         min(self.identity_manager.results[i][j]['cy'], self.identity_manager.results[i][j]['hy'], self.identity_manager.results[i][j]['by'])),
                         QtCore.QPointF(max(self.identity_manager.results[i][j]['cx'], self.identity_manager.results[i][j]['hx'], self.identity_manager.results[i][j]['bx']),
                         max(self.identity_manager.results[i][j]['cy'], self.identity_manager.results[i][j]['hy'], self.identity_manager.results[i][j]['by'])))
        for k in range(j + 1, self.identity_manager.ant_num):
            ant_two = QtCore.QRectF(QtCore.QPointF(min(self.identity_manager.results[i][k]['cx'], self.identity_manager.results[i][k]['hx'], self.identity_manager.results[i][k]['bx']),
                         min(self.identity_manager.results[i][k]['cy'], self.identity_manager.results[i][k]['hy'], self.identity_manager.results[i][k]['by'])),
                         QtCore.QPointF(max(self.identity_manager.results[i][k]['cx'], self.identity_manager.results[i][k]['hx'], self.identity_manager.results[i][k]['bx']),
                         max(self.identity_manager.results[i][k]['cy'], self.identity_manager.results[i][k]['hy'], self.identity_manager.results[i][k]['by'])))
            if ant_one.intersects(ant_two):
                self.add_fault(i, 'overlap', {j, k})

    #Main methods
    def init(self):
        for test in self.inits:
            test()

    def init_run(self):
        self.frame_count = 0
        for i in self.identity_manager.results:
            self.frame_count += 1
            for j in range(self.identity_manager.ant_num):
                for test in self.init_runs:
                    test(i, j)

    def process_init_run(self):
        for test in self.process_init_runs:
            test()

    def decision_run(self):
        for i in self.identity_manager.results:
            for j in range(self.identity_manager.ant_num):
                for test in self.decision_tests:
                    test(i, j)

    def compute_faults(self):
        self.faults = dict()
        self.switch_faults = dict()
        self.non_switch_faults = dict()
        self.fault_num = 0
        self.init()
        self.init_run()
        self.process_init_run()
        self.decision_run()
        return self.faults, self.switch_faults, self.non_switch_faults

    #Others
    def set_tolerances(self, len_tolerance=2, min_certainty=.2, proximity_tolerance=5, angular_tolerance=45):
        self.len_tolerance = len_tolerance
        self.min_certainty = min_certainty
        self.proximity_tolerance = proximity_tolerance
        self.angular_tolerance = angular_tolerance

    def angle(self, v_one, v_two):
        v_one_u = v_one / numpy.linalg.norm(v_one)
        v_two_u = v_two / numpy.linalg.norm(v_two)
        angle = numpy.arccos(numpy.dot(v_one_u, v_two_u))
        if numpy.isnan(angle):
            if (v_one_u == v_two_u).all():
                return 0.0
            else:
                return numpy.pi
        return angle

    def add_fault(self, frame_number, cause, involved_ants):
        self.fault_num += 1
        if cause in self.switch_causes:
            if frame_number not in self.switch_faults:
                self.switch_faults[frame_number] = []
        else:
            if frame_number not in self.non_switch_faults:
                self.non_switch_faults[frame_number] = []
        if frame_number not in self.faults:
            self.faults[frame_number] = []

        if not (type(involved_ants) is set):
            involved_ants = {involved_ants}

        fault = {'cause': cause, 'ants': involved_ants}

        if cause in self.switch_causes:
            self.switch_faults[frame_number].append(fault)
        else:
            self.non_switch_faults[frame_number].append(fault)
        self.faults[frame_number].append(fault)

    def combined_proximity(self, frame_no, ant_one, ant_two):
        prox_one = self.identity_manager.compute_distance(frame_no, ant_one, ant_two, 'hx', 'hy', 'hx', 'hy') + \
            self.identity_manager.compute_distance(frame_no, ant_one, ant_two, 'bx', 'by', 'bx', 'by')
        prox_two = self.identity_manager.compute_distance(frame_no, ant_one, ant_two, 'hx', 'hy', 'bx', 'by') + \
            self.identity_manager.compute_distance(frame_no, ant_one, ant_two, 'bx', 'by', 'hx', 'hy')
        return min(prox_one, prox_two) + self.identity_manager.compute_distance(frame_no, ant_one, ant_two, 'cx', 'cy', 'cx', 'cy')

    def proximity(self, frame_no, ant_one, ant_two):
        minimum = maxint
        for dx1, dy1 in [('cx', 'cy'), ('bx', 'by'), ('hx', 'hy')]:
            for dx2, dy2 in [('cx', 'cy'), ('bx', 'by'), ('hx', 'hy')]:
                distance = self.identity_manager.compute_distance(frame_no, ant_one, ant_two, dx1, dy1, dx2, dy2)
                if distance < minimum:
                    minimum = distance
        return minimum


    # An impossibly slow and painful test:
    # def better_overlap_test_decision(self, i, j):
    # 	length = 5
    # 	head = numpy.array([self.identity_manager.results[i][j]['hx'], self.identity_manager.results[i][j]['hy']])
    # 	tail = numpy.array([self.identity_manager.results[i][j]['bx'], self.identity_manager.results[i][j]['by']])
    # 	shape_one = self.get_bounding_shape(head, tail, length)
    # 	for k in range(j + 1, self.identity_manager.ant_num):
    # 		head = numpy.array([self.identity_manager.results[i][k]['hx'], self.identity_manager.results[i][k]['hy']])
    # 		tail = numpy.array([self.identity_manager.results[i][k]['bx'], self.identity_manager.results[i][k]['by']])
    # 		shape_two = self.get_bounding_shape(head, tail, length)
    # 		if self.intersects(shape_one, shape_two):
    # 			self.add_fault(i, 'overlap', {j, k})
    #
    # def get_bounding_shape(self, head, tail, length):
    # 	vector = (head - tail) * length / numpy.linalg.norm(head - tail)
    # 	normal = numpy.array([vector[1], - vector[0]])
    # 	bounding_shape = [
    # 		head + vector,
    # 		tail - vector,
    # 		head + normal,
    # 		head - normal,
    # 		tail + normal,
    # 		tail - normal
    # 	]
    # 	return bounding_shape
    #
    # def intersects(self, shape_one, shape_two):
    # 	for line_one in itertools.permutations(shape_one, 2):
    # 		for line_two in itertools.permutations(shape_two, 2):
    # 			if self.intersect_lines(line_one, line_two):
    # 				return True
    # 	return False
    #
    # def intersect_lines(self, line_one, line_two):
    # 	detA1 = numpy.linalg.det([
    # 		[line_one[0][0] - line_two[0][0], line_one[1][0] - line_two[0][0]],
    # 		[line_one[0][1] - line_two[0][1], line_one[1][1] - line_two[0][1]]
    # 	])
    # 	detA2 = numpy.linalg.det([
    # 		[line_two[0][0] - line_one[0][0], line_two[1][0] - line_one[0][0]],
    # 		[line_two[0][1] - line_one[0][1], line_two[1][1] - line_one[0][1]]
    # 	])
    # 	detB1 = numpy.linalg.det([
    # 		[line_one[0][0] - line_two[1][0], line_one[1][0] - line_two[1][0]],
    # 		[line_one[0][1] - line_two[1][1], line_one[1][1] - line_two[1][1]]
    # 	])
    # 	detB2 = numpy.linalg.det([
    # 		[line_two[0][0] - line_one[1][0], line_two[1][0] - line_one[1][0]],
    # 		[line_two[0][1] - line_one[1][1], line_two[1][1] - line_one[1][1]]
    # 	])
    # 	return detA1*detB1 < 0 and detA2*detB2 < 0