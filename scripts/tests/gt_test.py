from core.project.project import Project
from gui.statistics.region_reconstruction import get_trajectories
from utils.clearmetrics import ClearMetrics


def test_project(gt_measurements, test_measurements, frames, threshold):
    clear = ClearMetrics(gt_measurements, test_measurements, threshold)
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
    sn_id = 2
    name = 'Cam1'
    wd = '/Users/flipajs/Documents/wd/GT/'
    snapshot = {'chm': wd+name+'/.auto_save/'+str(sn_id)+'__chunk_manager.pkl',
                'gm': wd+name+'/.auto_save/'+str(sn_id)+'__graph_manager.pkl'}

    frames = range(5)
    p_test = Project()
    p_test.load(wd+name+'/cam1.fproj')
    gt_mesurements = get_trajectories(p_test, frames)

    del p_test

    p_gt = Project()
    p_gt.load(wd+name+'/cam1.fproj')
    test_mesurements = get_trajectories(p_gt, frames)

    test_project(
        gt_mesurements,
        test_mesurements,
        frames,
        1
                 )