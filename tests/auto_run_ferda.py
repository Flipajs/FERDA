__author__ = 'flipajs'
import sys
from gui.init_window import init_window
from PyQt4 import QtCore, QtGui
import pickle
from utils import misc
from utils.clearmetrics import clearmetrics
from scripts._clearmetrics import clearmetrics2
import copy
import numpy as np

from scripts.trajectories_data import eight_gt

CLEAR_PRECISION = 7
GT_PATH = '../data/sequences_data/gt'
STABLE_PATH = '../data/sequences_data/thesis_ferda_results'
RESULTS_PATH = '../data/sequences_data/_results'
NEW_DATA_PATH = '/home/flipajs/dump/'
FORMAT_UP_TO = 13
FORMAT_FLOAT_PRECISION = 5
FORMAT_UP_TO_DICT = 8
FORMAT_UP_TO_LIST = 30

def set_show_nothing():
    settings = QtCore.QSettings("FERDA")
    settings.setValue("show_mser_collection", False)
    settings.setValue("show_ants_collection", False)
    settings.setValue("show_image", False)
    settings.setValue("show_assignment_problem", False)

def run_ferda_with_params(params):
    settings = QtCore.QSettings("FERDA")
    settings.setValue("auto_run", True)
    #settings.setValue("")

    set_show_nothing()
    ex = init_window.InitWindow()

def compare_results(sequences, out_path):
    for seq in sequences:
        if seq == 'eight':
            analyse_results(seq, out_path)


def xy_from_ferda_out(data):
    """ from data saved by ferda algorithm exports x,y and
    returns them in dict = {0: [ant1, ant2, .... ]}
    where for each ant there is list [x, y]
    """
    xy = {}

    frames = len(data[0]['cx'])
    for i in range(frames):
        frame_data = []
        for a_id in range(len(data)):
            x = data[a_id]['cx'][i]
            y = data[a_id]['cy'][i]
            frame_data.append([x, y])

        xy[i] = frame_data

    return xy


def mismatches_in_frames(matches):
    mismatches = {}
    for key in matches:
        mismatches_num = matches[key].count(-1)
        if mismatches_num > 0:
            mismatches[key] = mismatches_num

    return  mismatches


def metrics_results(gt, data, old):
    if old:
        metrics = clearmetrics2.ClearMetrics(gt, data, CLEAR_PRECISION)
    else:
        metrics = clearmetrics.ClearMetrics(gt, data, CLEAR_PRECISION)

    metrics.match_sequence()

    problematic_frames = 'undefined'
    if not old:
        problematic_frames = metrics.get_problematic_frames()

    evaluation = {
        'mota': metrics.get_mota(),
        'motp': metrics.get_motp(),
        'fn': metrics.get_fn_count(),
        'fp': metrics.get_fp_count(),
        'mismatches': metrics.get_mismatches_count(),
        'object_count': metrics.get_object_count(),
        'matches_count': metrics.get_matches_count(),
        'problematic_frames': problematic_frames
    }

    return evaluation


def get_stable_results(seq):
    path = RESULTS_PATH+"/"+seq+".arr"
    try:
        f = open(path, 'rb')
        results = pickle.load(f)
    except IOError:
        gt_path = GT_PATH+"/"+seq+".arr"
        gt_data = misc.get_pickle_data(gt_path)

        stable_path = STABLE_PATH+"/"+seq+".arr"
        stable_data = misc.get_pickle_data(stable_path)

        results = metrics_results(gt_data, stable_data)

        f = open(path, 'wb')
        pickle.dump(results, f)

    return results


def get_stable_results(seq, old=False):
    old_suffix = ''
    if old:
        old_suffix = '_old'
    path = RESULTS_PATH+"/"+seq+old_suffix+".arr"
    try:
        f = open(path, 'rb')
        results = pickle.load(f)
    except IOError:
        gt_path = GT_PATH+"/"+seq+".arr"
        gt_data = misc.get_pickle_data(gt_path)

        stable_path = STABLE_PATH+"/"+seq+".arr"
        stable_data = misc.get_pickle_data(stable_path)

        results = metrics_results(gt_data, stable_data, old)

        f = open(path, 'wb')
        pickle.dump(results, f)

    return results


def print_frame_dict(stable, new):
    keys_s = stable.keys()
    keys_n = new.keys()
    keys = list(set(keys_s + keys_n))
    keys.sort()

    print misc.fill_spaces_up_to("FRAME", FORMAT_UP_TO_DICT, True), \
        misc.fill_spaces_up_to("  STABLE", FORMAT_UP_TO_LIST), \
        misc.fill_spaces_up_to(" NEW", FORMAT_UP_TO_LIST)

    i = 1
    divider_after_lines = 5
    for key in keys:
        frame = misc.fill_spaces_up_to(str(key), FORMAT_UP_TO_DICT, True)
        try:
            s = stable[key]
        except KeyError:
            s = ""

        try:
            n = new[key]
        except KeyError:
            n = ""

        s = misc.fill_spaces_up_to(str(s), FORMAT_UP_TO_LIST)
        n = misc.fill_spaces_up_to(str(n), FORMAT_UP_TO_LIST)

        print frame, ":", s + n

        if i % divider_after_lines == 0:
            divider = '.'*(2 + FORMAT_UP_TO_DICT + 2 * FORMAT_UP_TO_LIST)
            print divider

        i += 1


def print_problematic_frames(stable, new):
    print "MISMATCHES: "
    print_frame_dict(stable['mismatches'], new['mismatches'])
    print "\nFP: "
    print_frame_dict(stable['fp'], new['fp'])
    print "\nFN: "
    print_frame_dict(stable['fn'], new['fn'])


def analyse_results(stable_results, new_results):
    print misc.fill_spaces_up_to("KEY", FORMAT_UP_TO), \
        misc.fill_spaces_up_to("DIFF", FORMAT_UP_TO), \
        misc.fill_spaces_up_to("STABLE", FORMAT_UP_TO),\
        misc.fill_spaces_up_to("NEW", FORMAT_UP_TO)

    for key in stable_results:
        if key == 'problematic_frames':
            continue

        s = stable_results[key]
        n = new_results[key]
        diff = s-n

        if type(s) == float or type(s) == np.float64:
            s = misc.float2str(s, FORMAT_FLOAT_PRECISION)
            n = misc.float2str(n, FORMAT_FLOAT_PRECISION)
            diff = misc.float2str(diff, FORMAT_FLOAT_PRECISION)

        print misc.fill_spaces_up_to(str(key), FORMAT_UP_TO),\
            misc.fill_spaces_up_to(str(diff), FORMAT_UP_TO), \
            misc.fill_spaces_up_to(str(s), FORMAT_UP_TO), \
            misc.fill_spaces_up_to(str(n), FORMAT_UP_TO)

    print "\n\n"
    print "PROBLEMATIC FRAMES:"

    test = copy.deepcopy(stable_results['problematic_frames'])
    test['mismatches'][13] = [1, 3]

    print stable_results['problematic_frames']['mismatches']
    print test['mismatches']
    print_problematic_frames(
        stable_results['problematic_frames'],
        test
    )

    return


def main():
    results = get_stable_results('eight')
    results2 = get_stable_results('eight', old=True)

    analyse_results(results, results2)

if __name__ == '__main__':
    main()
