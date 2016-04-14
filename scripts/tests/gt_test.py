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
    print "One object swapped with None (exp: 1)"
    gt = {
        0: [15],
        1: [15]
    }
    ms = {
        0: [15, None],
        1: [None, 15]
    }
    return gt, ms


def data2():
    print "Two objects swapped with each other (exp: 1)"
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
    print "Three objects swapped with each other (exp: 3)"
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
    print "Four objects, two independent swaps (exp: 4)"
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


def data5():
    print "Two objects swapped and one of them lost (exp: 1)"
    gt = {
        0: [1, 2],
        1: [1, 2]
    }
    ms = {
        0: [1, 2],
        1: [2, None]
    }
    return gt, ms


def data6():
    print "Three objects, two swapped with each other, one with None (exp: 3)"
    gt = {
        0: [15, 5, 0],
        1: [15, 5, 0],
        2: [15, 5, 0]
    }
    ms = {
        0: [15, None, 5, 0],
        1: [15, None, 5, 0],
        2: [None, 15, 0, 5]
    }
    return gt, ms


if __name__ == "__main__":
    threshold = 0
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
    print "-----"
    data = data5()
    test_swaps(data[0], data[1], threshold)
    print "-----"
    data = data6()
    test_swaps(data[0], data[1], threshold)
    print "Done (%5.3f s)" % (time.time() - t)
