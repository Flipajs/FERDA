import time
from utils.clearmetrics.clearmetrics import ClearMetrics


def test_project(gt_measurements, test_measurements, threshold):
    clear = ClearMetrics(gt_measurements, test_measurements, threshold)
    clear.match_sequence()
    evaluation = [clear.get_mota(),
                  clear.get_motp(),
                  clear.get_fn_count(),
                  clear.get_fp_count(),
                  clear.get_mismatches_count(),
                  clear.get_mismatches_count_ignore_swaps(),
                  clear.get_object_count(),
                  clear.get_matches_count()]
    print evaluation


def test_swaps(gt_measurements, test_measurements, threshold):
    clear = ClearMetrics(gt_measurements, test_measurements, threshold)
    clear.match_sequence()
    print "Old mismatches: %s" % clear.get_mismatches_count()
    print "New mismatches: %s" % clear.get_mismatches_count_ignore_swaps()


def data1():
    gt = {
        0: [15],
        1: [15],
        2: [15]
    }
    ms = {
        0: [15, None],
        1: [15, None],
        2: [None, 15]
    }
    return gt, ms


def data2():
    gt = {
        0: [15, 10],
        1: [15, 10],
        2: [15, 10]
    }
    ms = {
        0: [15, 10],
        1: [15, 10],
        2: [10, 15]
    }
    return gt, ms


def data3():
    gt = {
        0: [15, 10, 5],
        1: [15, 10, 5],
        2: [15, 10, 5]
    }
    ms = {
        0: [15, 10, 5],
        1: [15, 10, 5],
        2: [10, 5, 15]
    }
    return gt, ms


def data4():
    gt = {
        0: [15, 10, 5, 0],
        1: [15, 10, 5, 0],
        2: [15, 10, 5, 0]
    }
    ms = {
        0: [15, 10, 5, 0],
        1: [15, 10, 5, 0],
        2: [10, 15, 0, 5]
    }
    return gt, ms


if __name__ == "__main__":
    threshold = 1
    t = time.time()
    print "Checking..."
    data = data1()
    test_swaps(data[0], data[1], threshold)
    print "-----"
    data = data2()
    test_swaps(data[0], data[1], threshold)
    print "-----"
    data = data3()
    test_swaps(data[0], data[1], threshold)
    print "-----"
    data = data4()
    test_swaps(data[0], data[1], threshold)
    print "Done (%5.3f s)" % (time.time() - t)
