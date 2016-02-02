from core.project.project import Project
from utils.clearmetrics.clearmetrics import ClearMetrics
import cPickle as pickle
from gui.statistics.region_reconstruction import gt_trajectories


def test_project(gt_f, project_f, threshold):
    gt = None

    with open(gt_f, 'rb') as f:
        gt = pickle.load(f)

    p = Project()
    p.load(project_f)


    keys_ = map(int, gt.keys())
    keys_ = sorted(keys_)

    measurements = gt_trajectories(p, keys_)


    clear = ClearMetrics(gt, measurements, threshold)
    clear.match_sequence()
    evaluation = [clear.get_mota(),
              clear.get_motp(),
              clear.get_fn_count(),
              clear.get_fp_count(),
              clear.get_mismatches_count(),
              clear.get_object_count(),
              clear.get_matches_count()]


    print evaluation

if __name__ == "__main__":
    test_project(
        '/Users/flipajs/Documents/wd/GT/C210_GT/out_regions.pkl',
        '/Users/flipajs/Documents/wd/GT/C210_GT/C210.fproj',
        1
                 )