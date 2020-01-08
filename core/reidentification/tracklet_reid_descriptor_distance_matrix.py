"""
Computes a tracklet distance matrices based on reidentification descriptors.

For a visualization see the VAIB paper and experiments/tracking/180427_vaib_tracklet_reid_descriptor_distance_matrix.ipynb.
"""
from joblib import Parallel, delayed
from core.project.project import Project
from core.id_detection.learning_process import LearningProcess
from core.id_detection.complete_set_matching import CompleteSetMatching
from os.path import join
import numpy as np
from core.id_detection.complete_set_matching import get_probability_that_prototypes_are_same_tracks
import itertools
import tqdm
import utils.gt.mot as gt
import joblib


def prototype_distances(tracklets, prototypes):
    tracklet_prototypes_sorted = sorted(zip(tracklets, prototypes), key=lambda x: x[0].animal_id_)
    tracklet_prototypes_sorted_with_id = [tp for tp in tracklet_prototypes_sorted if tp[0].animal_id_ >= 0]
    
    tp = tracklet_prototypes_sorted_with_id
    num = len(tp)
    prob_ij = Parallel(n_jobs=-1, verbose=10)(delayed(get_probability_that_prototypes_are_same_tracks)(tp[i][1], tp[j][1]) for i, j in
                                              list(itertools.product(list(range(num)), list(range(num)))))
    m = np.zeros((num, num))
    for prob, (i, j) in zip(prob_ij, itertools.product(list(range(num)), list(range(num)))):
        m[i, j] = prob
    return m, tp


def get_distance_matrix(project_path, gt_path):
    p = Project(project_path)
    lp = LearningProcess(p)
    csm = CompleteSetMatching(p, lp, join(project_path, 'descriptors.pkl'), quality_threshold=0.2, quality_threshold2=0.01)
    ground_truth = gt.Mot()
    ground_truth.load(gt_path)
    
    tracklets = [t for t in p.chm.chunk_gen() if t.is_single()]
    prototypes = [csm.get_tracklet_prototypes(t) for t in tracklets]
    gt_matches = ground_truth.match_on_data(p, max_d=100)

    # add gt id to tracklets
    for t in tqdm.tqdm(tracklets):
        try:
            gt_ids = [gt_matches[frame].index(t.id()) for frame in range(t.start_frame(), t.end_frame())]
            if len(set(gt_ids)) == 1:
                t.animal_id_ = gt_ids[0]
        except ValueError:
            pass
    m, tp_sorted = prototype_distances(tracklets, prototypes)
    return {'matrix': m, 'tracklets_prototypes': tp_sorted}


if __name__ == '__main__':
    project_paths = [
        '/datagrid/ferda/projects/old/6_results_vaib_2018/Cam1_clip_arena_fixed',
        '/datagrid/ferda/projects/old/6_results_vaib_2018/Camera3-5min',
        '/datagrid/ferda/projects/old/6_results_vaib_2018/Sowbug3-fixed-segmentation',
        '/datagrid/ferda/projects/old/6_results_vaib_2018/5Zebrafish_nocover_22min',
    ]

    gt_paths = [
        'data/GT/Cam1_clip.avi.pkl',
        'data/GT/Camera3-5min.mp4.pkl',
        'data/GT/Sowbug3.pkl',
        'data/GT/5Zebrafish_nocover_22min.pkl',
    ]

    matrices = []
    for project_path, gt_path in zip(project_paths, gt_paths):
        m = get_distance_matrix(project_path, gt_path)
        m['project_path'] = project_path
        matrices.append(m)
        joblib.dump(matrices, 'distance_matrices.gz')

